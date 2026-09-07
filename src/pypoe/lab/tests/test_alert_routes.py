"""Unit tests for ``pypoe.lab.alert_routes``.

Uses FastAPI's TestClient + monkeypatches the Slack post + claude
subprocess so the test never touches the network or the shell.
"""

from __future__ import annotations

import asyncio
import json

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from pypoe.lab import alert_routes
from pypoe.lab.config import reload_config
from pypoe.lab.http_client import LabClient


def _mk_app(monkeypatch, *, claude_output: str = "Investigation summary"):
    posted: list[dict] = []
    investigations: list[str] = []

    async def fake_post_slack(channel, text, thread_ts=None):
        posted.append(
            {"channel": channel, "text": text, "thread_ts": thread_ts}
        )
        return f"ts-{len(posted)}"

    async def fake_run_claude(prompt):
        investigations.append(prompt)
        return claude_output

    monkeypatch.setattr(alert_routes, "_post_slack", fake_post_slack)
    monkeypatch.setattr(alert_routes, "_run_claude", fake_run_claude)

    fastapi_app = FastAPI()
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={}))
    lab = LabClient(
        base_url="http://test",
        client=httpx.AsyncClient(base_url="http://test", transport=transport),
    )
    alert_routes.register_alert_routes(
        fastapi_app, client=lab, max_concurrent=2, slack_channel="#test"
    )
    return fastapi_app, posted, investigations


def test_build_prompt_with_models_lists_each_model():
    """The synthesised prompt MUST mention every configured consult model."""
    prompt = alert_routes._build_investigation_prompt(
        monitor="plateloc",
        msg="No response from COM3",
        consult_models=("GPT-5.4", "Claude-Opus-4.7", "Gemini-3.1-Pro"),
    )
    assert "plateloc" in prompt
    assert "No response from COM3" in prompt
    # Each model name appears in the listed block.
    for m in ("GPT-5.4", "Claude-Opus-4.7", "Gemini-3.1-Pro"):
        assert m in prompt
    # Consult-block instructions present.
    assert "consult_poe" in prompt
    assert "synthesised" in prompt or "synthesis" in prompt.lower()
    # Don't propose calling /control/*.
    assert "do not propose" in prompt.lower()


def test_prompts_ask_to_read_prior_observations():
    """Both prompt shapes must steer the investigator to read prior agent
    findings and journal a matchable headline, so recurrences build on the
    last root cause instead of starting cold."""
    device = alert_routes._build_device_prompt(
        device_id="ot2_hte", event="unreachable", msg="disconnected",
        consult_models=("GPT-5.4",),
    )
    kuma = alert_routes._build_investigation_prompt(
        monitor="aggregator", msg="down", consult_models=(),
    )
    for prompt in (device, kuma):
        assert "recent_observations" in prompt
        assert "recurrence" in prompt.lower()
        assert "headline" in prompt.lower()


def test_build_prompt_no_models_uses_solo_block():
    """Empty model list → no consult step, no synthesis block."""
    prompt = alert_routes._build_investigation_prompt(
        monitor="aggregator",
        msg="down",
        consult_models=(),
    )
    assert "Skip Poe consultation" in prompt
    # Solo tail = 2-4 line summary instruction, no synthesis bullet rules.
    assert "consult_poe" not in prompt
    assert "synthesis" not in prompt.lower()


def test_kuma_webhook_down_posts_investigating_then_summary(monkeypatch):
    app, posted, investigations = _mk_app(
        monkeypatch, claude_output="Summary lines here"
    )

    with TestClient(app) as client:
        resp = client.post(
            "/alerts/kuma",
            json={
                "heartbeat": {"status": 0, "msg": "connection refused"},
                "monitor": {"name": "aggregator"},
                "msg": "connection refused",
            },
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["action"] == "investigating"
    assert body["monitor"] == "aggregator"

    # First post is the rotating-light line.
    assert posted, "expected at least one Slack post"
    assert "Investigating" in posted[0]["text"]
    assert posted[0]["channel"] == "#test"
    assert posted[0]["thread_ts"] is None

    # The background task should have run after the TestClient context
    # closed (which awaits all pending tasks). The captured value is the
    # fully built prompt, which must carry the monitor, the message, and
    # the default consult models (ConsultSection — see pypoe.lab.config).
    assert len(investigations) == 1
    prompt = investigations[0]
    assert "aggregator" in prompt
    assert "connection refused" in prompt
    # Default consult models (ConsultSection — see pypoe.lab.config). Derived
    # rather than hardcoded so changing the catalog can't leave this asserting
    # models nobody can reach any more.
    from pypoe.lab.config import ConsultSection

    for model in ConsultSection().models:
        assert model in prompt
    assert len(posted) == 2
    threaded = posted[1]
    assert threaded["thread_ts"] == "ts-1"  # threaded under the first message
    assert "Summary lines here" in threaded["text"]


def test_kuma_webhook_recovery_posts_recovery_only(monkeypatch):
    app, posted, investigations = _mk_app(monkeypatch)

    with TestClient(app) as client:
        resp = client.post(
            "/alerts/kuma",
            json={
                "heartbeat": {"status": 1, "msg": "back up"},
                "monitor": {"name": "aggregator"},
                "msg": "back up",
            },
        )
    assert resp.status_code == 202
    assert resp.json()["action"] == "recovered"
    assert len(posted) == 1
    assert "recovered" in posted[0]["text"]
    # No investigation kicked off on recovery.
    assert investigations == []


def test_kuma_webhook_truncates_long_claude_output(monkeypatch):
    long_text = "x" * 5000
    app, posted, _ = _mk_app(monkeypatch, claude_output=long_text)
    with TestClient(app) as client:
        client.post(
            "/alerts/kuma",
            json={
                "heartbeat": {"status": 0, "msg": "down"},
                "monitor": {"name": "any"},
            },
        )
    # Two posts; second is the threaded reply, which must be truncated.
    threaded = posted[1]
    assert "(truncated)" in threaded["text"]
    assert len(threaded["text"]) <= alert_routes._MAX_CLAUDE_OUTPUT_CHARS


def test_device_alert_down_posts_investigating_then_summary(monkeypatch):
    app, posted, investigations = _mk_app(
        monkeypatch, claude_output="Device diagnosis"
    )

    with TestClient(app) as client:
        resp = client.post(
            "/alerts/device",
            json={
                "device_id": "plateloc",
                "event": "error",
                "state": "error",
                "message": "COM timeout",
                "last_error": {"code": "com_other", "message": "COM timeout"},
            },
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["action"] == "investigating"
    assert body["device"] == "plateloc"

    assert "plateloc" in posted[0]["text"]
    assert "ERROR" in posted[0]["text"]
    assert "Investigating" in posted[0]["text"]

    # Prompt is device-focused: targets the device, carries last_error.
    assert len(investigations) == 1
    prompt = investigations[0]
    assert 'get_equipment_status("plateloc")' in prompt
    assert "com_other" in prompt

    threaded = posted[1]
    assert threaded["thread_ts"] == "ts-1"
    assert "Device diagnosis" in threaded["text"]


def test_device_alert_storm_mentions_other_devices(monkeypatch):
    app, posted, investigations = _mk_app(monkeypatch)

    with TestClient(app) as client:
        resp = client.post(
            "/alerts/device",
            json={
                "device_id": "ot2_hte",
                "event": "unreachable",
                "message": "3 devices down in one sweep",
                "devices": ["plateloc", "cytation_5"],
            },
        )
    assert resp.status_code == 202
    assert "(+2 more)" in posted[0]["text"]
    prompt = investigations[0]
    assert "plateloc" in prompt and "cytation_5" in prompt


def test_device_alert_prefixes_platform_and_injects_context(monkeypatch):
    """When the device is in a platform section, the Slack headline is
    prefixed with the platform and the prompt carries the co-located devices
    for shared-cause reasoning."""
    posted: list[dict] = []
    investigations: list[str] = []

    async def fake_post_slack(channel, text, thread_ts=None):
        posted.append({"text": text, "thread_ts": thread_ts})
        return f"ts-{len(posted)}"

    async def fake_run_claude(prompt):
        investigations.append(prompt)
        return "diagnosis"

    monkeypatch.setattr(alert_routes, "_post_slack", fake_post_slack)
    monkeypatch.setattr(alert_routes, "_run_claude", fake_run_claude)

    def handler(request):
        if request.url.path == "/api/platforms":
            return httpx.Response(
                200,
                json={
                    "sections": [
                        {
                            "id": "hte",
                            "title": "HTE Platform",
                            "equipment": ["ot2_hte", "plateloc", "cytation_5"],
                        }
                    ]
                },
            )
        return httpx.Response(200, json={})

    lab = LabClient(
        base_url="http://test",
        client=httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ),
    )
    app = FastAPI()
    alert_routes.register_alert_routes(
        app, client=lab, max_concurrent=2, slack_channel="#test"
    )

    with TestClient(app) as client:
        resp = client.post(
            "/alerts/device",
            json={"device_id": "ot2_hte", "event": "unreachable", "message": "gone"},
        )
    assert resp.status_code == 202
    # Headline carries the platform label.
    assert "HTE Platform · ot2_hte" in posted[0]["text"]
    # Prompt names the platform and its co-located (sibling) devices.
    prompt = investigations[0]
    assert "Platform: HTE Platform" in prompt
    assert "plateloc" in prompt and "cytation_5" in prompt


def test_device_alert_unknown_platform_falls_back_to_bare_id(monkeypatch):
    """A device not in any section keeps today's bare-id behaviour."""
    app, posted, investigations = _mk_app(monkeypatch)  # mock returns {} for /api/platforms
    with TestClient(app) as client:
        client.post(
            "/alerts/device",
            json={"device_id": "some_service", "event": "error", "message": "x"},
        )
    assert "*some_service*" in posted[0]["text"]
    assert "·" not in posted[0]["text"]  # no platform separator


def test_device_alert_recovery_posts_recovery_only(monkeypatch):
    app, posted, investigations = _mk_app(monkeypatch)

    with TestClient(app) as client:
        resp = client.post(
            "/alerts/device",
            json={
                "device_id": "plateloc",
                "event": "recovered",
                "message": "back to ready",
            },
        )
    assert resp.status_code == 202
    assert resp.json()["action"] == "recovered"
    assert len(posted) == 1
    assert "recovered" in posted[0]["text"]
    assert investigations == []


def test_device_alert_rejects_unknown_event(monkeypatch):
    app, posted, investigations = _mk_app(monkeypatch)

    with TestClient(app) as client:
        resp = client.post(
            "/alerts/device",
            json={"device_id": "plateloc", "event": "exploded"},
        )
    assert resp.status_code == 422
    assert posted == []
    assert investigations == []


def test_concurrency_bound_is_respected(monkeypatch):
    """Two simultaneous alerts shouldn't run more than `max_concurrent` claude
    subprocesses in parallel. Verified via observable peak concurrency."""

    async def main():
        peak = {"value": 0, "current": 0, "lock": asyncio.Lock()}
        ready = asyncio.Event()
        release = asyncio.Event()

        async def fake_post_slack(channel, text, thread_ts=None):
            return "ts"

        async def fake_run_claude(prompt):
            async with peak["lock"]:
                peak["current"] += 1
                peak["value"] = max(peak["value"], peak["current"])
                if peak["current"] >= 2:
                    ready.set()
            await release.wait()
            async with peak["lock"]:
                peak["current"] -= 1
            return "done"

        monkeypatch.setattr(alert_routes, "_post_slack", fake_post_slack)
        monkeypatch.setattr(alert_routes, "_run_claude", fake_run_claude)

        fastapi_app = FastAPI()
        transport = httpx.MockTransport(
            lambda r: httpx.Response(200, json={})
        )
        lab = LabClient(
            base_url="http://test",
            client=httpx.AsyncClient(base_url="http://test", transport=transport),
        )
        alert_routes.register_alert_routes(
            fastapi_app, client=lab, max_concurrent=2, slack_channel="#test"
        )

        from fastapi.testclient import TestClient

        with TestClient(fastapi_app) as tc:
            for _ in range(4):
                tc.post(
                    "/alerts/kuma",
                    json={
                        "heartbeat": {"status": 0, "msg": "down"},
                        "monitor": {"name": "x"},
                    },
                )
            # Give the background tasks a moment to start; release them
            # so the TestClient teardown can finish.
            try:
                await asyncio.wait_for(ready.wait(), timeout=2.0)
            finally:
                release.set()
        return peak["value"]

    peak = asyncio.run(main())
    assert peak <= 2, f"peak concurrency {peak} exceeded max_concurrent=2"


# ---------------------------------------------------------------------------
# _run_claude hardening (model pin, strict MCP config, cwd isolation, timeout)
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, out=b"ok summary", err=b"", returncode=0):
        self._out = out
        self._err = err
        self.returncode = returncode

    async def communicate(self):
        return (self._out, self._err)

    async def wait(self):
        return self.returncode


def test_run_claude_builds_hardened_command(monkeypatch, tmp_path):
    """The investigation subprocess pins a model, injects only the pypoe-lab
    MCP server strictly, runs outside the repo tree, and carries a system
    prompt — matching the dashboard assistant's safeguards."""

    captured: dict = {}

    async def fake_exec(*args, **kwargs):
        captured["args"] = list(args)
        captured["cwd"] = kwargs.get("cwd")
        return _FakeProc(out=b"root cause: air pressure")

    # Deterministic config (no packaged slack.yaml), isolated runtime dir.
    monkeypatch.setenv("PYPOE_LAB_CONFIG", str(tmp_path / "no-such.yaml"))
    monkeypatch.setenv("PYPOE_INVESTIGATOR_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("LAB_INVESTIGATION_MODEL", "sonnet")
    reload_config()
    monkeypatch.setattr(alert_routes, "_claude_binary", lambda: "/usr/bin/claude")
    monkeypatch.setattr(alert_routes.asyncio, "create_subprocess_exec", fake_exec)

    try:
        out = asyncio.run(alert_routes._run_claude("investigate ot2_hte"))
    finally:
        reload_config()

    args = captured["args"]
    assert args[0] == "/usr/bin/claude"
    assert "-p" in args and "investigate ot2_hte" in args
    assert args[args.index("--model") + 1] == "sonnet"
    assert "--strict-mcp-config" in args
    assert "--append-system-prompt" in args
    assert args[args.index("--allowedTools") + 1] == "mcp__pypoe-lab__*"
    # cwd is the isolated runtime dir, not the repo tree.
    assert captured["cwd"] == str(tmp_path)
    # An explicit, strict mcp.json was written for exactly the pypoe-lab server.
    mcp_path = args[args.index("--mcp-config") + 1]
    assert mcp_path == str(tmp_path / "mcp.json")
    cfg = json.loads((tmp_path / "mcp.json").read_text())
    assert list(cfg["mcpServers"].keys()) == ["pypoe-lab"]
    assert cfg["mcpServers"]["pypoe-lab"]["args"] == ["lab-mcp"]
    assert out == "root cause: air pressure"


def test_run_claude_times_out_and_kills(monkeypatch, tmp_path):
    """A hung CLI is killed at the wallclock cap and reported, not left to
    linger against the concurrency budget."""

    killed = {"value": False}

    class _HangProc:
        returncode = None

        async def communicate(self):
            await asyncio.sleep(10)
            return (b"", b"")

        def kill(self):
            killed["value"] = True
            self.returncode = -9

        async def wait(self):
            return -9

    async def fake_exec(*args, **kwargs):
        return _HangProc()

    monkeypatch.setenv("PYPOE_LAB_CONFIG", str(tmp_path / "no-such.yaml"))
    monkeypatch.setenv("PYPOE_INVESTIGATOR_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("LAB_INVESTIGATION_TIMEOUT_S", "0.05")
    reload_config()
    monkeypatch.setattr(alert_routes, "_claude_binary", lambda: "/usr/bin/claude")
    monkeypatch.setattr(alert_routes.asyncio, "create_subprocess_exec", fake_exec)

    try:
        out = asyncio.run(alert_routes._run_claude("investigate"))
    finally:
        reload_config()

    assert "timeout" in out.lower()
    assert killed["value"] is True


# ---------------------------------------------------------------------------
# SDL Assistant self-healing monitor
# ---------------------------------------------------------------------------


def test_assistant_health_ok_parses_configured(monkeypatch):
    """configured:true + a model ⇒ healthy; non-200 / configured:false ⇒ down."""
    from pypoe.lab.http_client import LabClient

    async def testcases():
        cases = [
            (200, {"configured": True, "model": "m"}, True),
            (200, {"configured": False, "backend": "openai"}, False),
            (500, {}, False),
            (200, "not json", False),
        ]
        results = []
        for status, payload, expected in cases:
            transport = httpx.MockTransport(lambda r, _p=payload, _s=status: httpx.Response(_s, json=_p))
            lab = LabClient(
                base_url="http://t",
                client=httpx.AsyncClient(base_url="http://t", transport=transport),
            )
            ok, _det = await alert_routes._assistant_health_ok(lab, "http://t/api/x")
            results.append(ok == expected)
            await lab.aclose()
        return all(results)

    assert asyncio.run(testcases())


def test_assistant_health_transport_error_is_down(monkeypatch):
    from pypoe.lab.http_client import LabClient

    def boom(request):
        raise httpx.ConnectError("refused", request=request)

    transport = httpx.MockTransport(boom)
    lab = LabClient(
        base_url="http://t",
        client=httpx.AsyncClient(base_url="http://t", transport=transport),
    )
    ok, det = asyncio.run(alert_routes._assistant_health_ok(lab, "http://t/api/x"))
    assert ok is False
    assert "unreachable" in det
    asyncio.run(lab.aclose())


def test_assistant_key_present(monkeypatch, tmp_path):
    """Key present ⇒ True; missing/quoted-empty ⇒ False (no value leak)."""
    env = tmp_path / ".env"
    env.write_text("ASSISTANT_OPENAI_API_KEY=sk-12345\nOTHER=X\n")
    assert alert_routes._assistant_key_present(str(tmp_path)) == (True, "key present")

    env.write_text("OTHER=X\n")
    ok, det = alert_routes._assistant_key_present(str(tmp_path))
    assert ok is False
    assert "KEY" in det

    env.write_text('ASSISTANT_OPENAI_API_KEY=""\n')
    ok, _det = alert_routes._assistant_key_present(str(tmp_path))
    assert ok is False


def test_restart_service_kills_pid(monkeypatch):
    """Relaunch prefers SIGKILL of the MainPID (the sudo-less bounce)."""
    calls = {"kill": [], "restart": []}

    def fake_mainpid(service):
        return "12345"

    def fake_kill(*args, **kwargs):
        calls["kill"].append(args[0])
        import subprocess
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    def fake_systemctl_restart(*args, **kwargs):
        calls["restart"].append(args[0])
        import subprocess
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(alert_routes, "_service_mainpid", fake_mainpid)
    monkeypatch.setattr(alert_routes.subprocess, "run", fake_kill)
    ok, det = alert_routes._restart_service("ac-organic-lab-api.service")
    assert ok is True
    assert calls["kill"] == [["kill", "-KILL", "12345"]]
    assert calls["restart"] == []
    assert "12345" in det


def test_restart_service_falls_back_to_systemctl(monkeypatch):
    calls = {"run": []}

    def fake_mainpid(service):
        return None  # no resolvable PID → fall back

    def fake_run(*args, **kwargs):
        calls["run"].append(args[0])
        import subprocess
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(alert_routes, "_service_mainpid", fake_mainpid)
    monkeypatch.setattr(alert_routes.subprocess, "run", fake_run)
    ok, det = alert_routes._restart_service("svc")
    assert ok is True
    assert calls["run"][0] == ["systemctl", "restart", "svc"]


def test_assistant_remediate_restarts_and_recovers(monkeypatch):
    """Full remediate: down → service active → bounce → key next → healthy."""
    from pypoe.lab.config import AssistantSection, LabConfig

    cfg = LabConfig(
        assistant=AssistantSection(
            service_name="svc", env_root="/nowhere", restart_wait_s=0.0,
            confirm_wait_s=0.0,  # gate off: this test exercises the restart path
        )
    )
    rev = {"n": 0}

    async def fake_health(lab, url):
        rev["n"] += 1
        # Baseline probe fails; the probe after the "restart" succeeds.
        if rev["n"] >= 2:
            return True, "ok"
        return False, "down"

    monkeypatch.setattr(alert_routes, "_assistant_health_ok", fake_health)
    monkeypatch.setattr(alert_routes, "_service_active", lambda s: True)
    monkeypatch.setattr(
        alert_routes, "_restart_service", lambda s: (True, "bounced")
    )
    monkeypatch.setattr(
        alert_routes, "_assistant_key_present", lambda e: (True, "key present")
    )

    async def run():
        return await alert_routes._assistant_remediate(cfg, None)  # type: ignore[arg-type]

    recovered, steps = asyncio.run(run())
    assert recovered is True
    labels = [s[0] for s in steps]
    assert any("service svc active" in l for l in labels)
    assert any("restart svc" in l for l in labels)


def test_assistant_alert_route_down_posts_alert_and_report(monkeypatch):
    """POST /alerts/assistant on a DOWN assistant alerts + posts the report."""
    posted: list[dict] = []

    async def fake_post(channel, text, thread_ts=None):
        posted.append({"text": text, "thread_ts": thread_ts})
        return f"ts-{len(posted)}"

    async def fake_health(lab, url):
        return False, "configured=false"

    async def fake_remediate(cfg, lab):
        return False, [
            ("service svc active", True, "active"),
            ("restart svc", False, "nope"),
            ("ASSISTANT_OPENAI_API_KEY present", False, "missing"),
        ]

    monkeypatch.setattr(alert_routes, "_post_slack", fake_post)
    monkeypatch.setattr(alert_routes, "_assistant_health_ok", fake_health)
    monkeypatch.setattr(alert_routes, "_assistant_remediate", fake_remediate)

    appx = FastAPI()
    # monitor disabled keeps this deterministic (route only, no loop task)
    alert_routes.register_assistant_monitor(appx, slack_channel="#test")

    with TestClient(appx) as client:
        resp = client.post("/alerts/assistant")
    assert resp.status_code == 202
    assert resp.json()["action"] == "assistant_selfheal"

    assert len(posted) == 2
    assert "SDL Assistant" in posted[0]["text"]
    assert "trying fixes" in posted[0]["text"]
    assert posted[1]["thread_ts"] == "ts-1"  # threaded
    assert "STILL DOWN" in posted[1]["text"]
    assert "ASSISTANT_OPENAI_API_KEY present" in posted[1]["text"]


def test_assistant_alert_route_healthy_posts_nothing(monkeypatch):
    """A healthy assistant POSTs nothing (pure liveness/self-heal no-op)."""
    posted: list[dict] = []

    async def fake_post(channel, text, thread_ts=None):
        posted.append({"text": text})

    async def fake_health(lab, url):
        return True, "ok"

    monkeypatch.setattr(alert_routes, "_post_slack", fake_post)
    monkeypatch.setattr(alert_routes, "_assistant_health_ok", fake_health)

    appx = FastAPI()
    alert_routes.register_assistant_monitor(appx, slack_channel="#test")
    with TestClient(appx) as client:
        client.post("/alerts/assistant")
    assert posted == []


# ---------------------------------------------------------------------------
# SDL Dashboard surface monitor
# ---------------------------------------------------------------------------


class _FakeDashClient:
    """Minimal async client stub: statuses keyed by path suffix."""

    def __init__(self, statuses: dict[str, int]):
        self._statuses = statuses

    async def get(self, url: str):
        class _Resp:
            def __init__(self, code, body):
                self.status_code = code
                self._body = body

            def json(self):
                return self._body

        code = self._statuses.get(url.split("8001", 1)[-1], 500)
        return _Resp(code, {"ok": True})


def _dash_cfg(**kw):
    from pypoe.lab.config import DashboardSection, LabConfig

    return LabConfig(dashboard=DashboardSection(**kw))


def test_dashboard_ok_all_healthy(monkeypatch):
    lab = LabClient()
    lab._client = _FakeDashClient({"/api/openapi.json": 200, "/api/catalog": 200, "/api/equipment": 200})
    cfg = _dash_cfg()
    ok, det = asyncio.run(alert_routes._dashboard_ok(lab, cfg))
    assert ok is True
    assert "HTTP 200 ok" in det


def test_dashboard_ok_one_500_is_down(monkeypatch):
    lab = LabClient()
    lab._client = _FakeDashClient({"/api/openapi.json": 500, "/api/catalog": 200, "/api/equipment": 200})
    cfg = _dash_cfg()
    ok, det = asyncio.run(alert_routes._dashboard_ok(lab, cfg))
    assert ok is False
    assert "openapi.json -> HTTP 500" in det


def test_dashboard_url_joins_base_and_path():
    assert alert_routes._dashboard_url("http://127.0.0.1:8001/", "api/health") == \
        "http://127.0.0.1:8001/api/health"


def test_dashboard_remediate_restarts_and_recovers(monkeypatch):
    """Full remediate: down → service active → bounce restart → healthy."""
    # confirm gate off: this test exercises the restart path
    cfg = _dash_cfg(service_name="svc", restart_wait_s=0.0, confirm_wait_s=0.0)
    rev = {"n": 0}

    async def fake_ok(lab, cfg):
        rev["n"] += 1
        if rev["n"] >= 2:
            return True, "ok"
        return False, "HTTP 500"

    monkeypatch.setattr(alert_routes, "_dashboard_ok", fake_ok)
    monkeypatch.setattr(alert_routes, "_service_active", lambda s: True)
    monkeypatch.setattr(alert_routes, "_restart_service", lambda s: (True, "bounced"))

    async def run():
        return await alert_routes._dashboard_remediate(cfg, None)  # type: ignore[arg-type]

    recovered, steps = asyncio.run(run())
    assert recovered is True
    labels = [s[0] for s in steps]
    assert any("service svc active" in l for l in labels)
    assert any("restart svc" in l for l in labels)


def test_dashboard_alert_route_down_posts_alert_and_report(monkeypatch):
    """POST /alerts/dashboard on a DOWN dashboard alerts + posts the report."""
    posted: list[dict] = []

    async def fake_post(channel, text, thread_ts=None):
        posted.append({"text": text, "thread_ts": thread_ts})
        return f"ts-{len(posted)}"

    async def fake_ok(lab, cfg):
        return False, "openapi.json -> HTTP 500"

    async def fake_remediate(cfg, lab):
        return False, [
            ("service svc active", True, "active"),
            ("restart svc", False, "nope"),
        ]

    monkeypatch.setattr(alert_routes, "_post_slack", fake_post)
    monkeypatch.setattr(alert_routes, "_dashboard_ok", fake_ok)
    monkeypatch.setattr(alert_routes, "_dashboard_remediate", fake_remediate)

    appx = FastAPI()
    # monitor disabled keeps this deterministic (route only, no loop task)
    alert_routes.register_dashboard_monitor(appx, slack_channel="#test")

    with TestClient(appx) as client:
        resp = client.post("/alerts/dashboard")
    assert resp.status_code == 202
    assert resp.json()["action"] == "dashboard_selfheal"

    assert len(posted) == 2
    assert "SDL Dashboard" in posted[0]["text"]
    assert "trying fixes" in posted[0]["text"]
    assert posted[1]["thread_ts"] == "ts-1"
    assert "STILL DOWN" in posted[1]["text"]


def test_dashboard_alert_route_healthy_posts_nothing(monkeypatch):
    """A healthy dashboard POSTs nothing (pure liveness/self-heal no-op)."""
    posted: list[dict] = []

    async def fake_post(channel, text, thread_ts=None):
        posted.append({"text": text})

    async def fake_ok(lab, cfg):
        return True, "all ok"

    monkeypatch.setattr(alert_routes, "_post_slack", fake_post)
    monkeypatch.setattr(alert_routes, "_dashboard_ok", fake_ok)

    appx = FastAPI()
    alert_routes.register_dashboard_monitor(appx, slack_channel="#test")
    with TestClient(appx) as client:
        client.post("/alerts/dashboard")
    assert posted == []



# ---------------------------------------------------------------------------
# Monitor-loop latch pairing + the SIGKILL confirm gate (2026-08-30)
# ---------------------------------------------------------------------------
# gaia's memory thrash froze every process ~25 s at a time; single lost probes
# then produced ':white_check_mark: recovered.' lines with no outage ever
# posted, and longer stalls got the perfectly-healthy API SIGKILLed. These
# tests pin the two fixes: `was_down` means "a DOWN alert stands unanswered",
# and the remediation bounce is gated behind a confirm re-probe.


class _StopLoop(Exception):
    """Raised by the probe fake to break out of the infinite monitor loop."""


def _assistant_cfg(**kw):
    from pypoe.lab.config import AssistantSection, LabConfig

    return LabConfig(assistant=AssistantSection(**kw))


def _run_assistant_loop(monkeypatch, cfg, probes, handle_recovered):
    """Drive `_assistant_monitor_loop` over a finite probe sequence.

    Returns the ordered (kind, detail) events posted. `probe_interval_s=0`
    makes the loop's real `asyncio.sleep(0)` a plain yield, so no sleeping.
    """
    events: list[tuple[str, str | None]] = []
    seq = iter(probes)

    async def fake_probe(lab, url):
        try:
            return next(seq)
        except StopIteration:
            raise _StopLoop()

    async def fake_handle(channel, lab, c, det):
        events.append(("down", det))
        return handle_recovered

    async def fake_recover(channel):
        events.append(("recovered", None))

    monkeypatch.setattr(alert_routes, "_assistant_health_ok", fake_probe)
    monkeypatch.setattr(alert_routes, "_assistant_handle_down", fake_handle)
    monkeypatch.setattr(alert_routes, "_assistant_recover", fake_recover)

    async def run():
        await alert_routes._assistant_monitor_loop(
            "#test", None, cfg, asyncio.Semaphore(1)
        )

    with pytest.raises(_StopLoop):
        asyncio.run(run())
    return events


def test_monitor_transient_down_posts_no_orphan_recovery(monkeypatch):
    """A DOWN the remediation immediately resolved must clear the latch: the
    old code left it armed and posted an orphan 'recovered.' next tick."""
    cfg = _assistant_cfg(probe_interval_s=0.0, failures_to_alert=1)
    events = _run_assistant_loop(
        monkeypatch,
        cfg,
        probes=[(False, "timeout"), (True, "ok")],
        handle_recovered=True,  # remediation report ended on the green line
    )
    assert events == [("down", "timeout")]


def test_monitor_unresolved_down_still_pairs_with_recovery(monkeypatch):
    """When remediation could NOT recover it, the latch stays armed and the
    eventual healthy probe posts exactly one recovery line."""
    cfg = _assistant_cfg(probe_interval_s=0.0, failures_to_alert=1)
    events = _run_assistant_loop(
        monkeypatch,
        cfg,
        probes=[(False, "HTTP 500"), (True, "ok")],
        handle_recovered=False,
    )
    assert events == [("down", "HTTP 500"), ("recovered", None)]


def test_monitor_single_failure_below_threshold_posts_nothing(monkeypatch):
    """failures_to_alert damps one lost probe (a machine stall) entirely."""
    cfg = _assistant_cfg(probe_interval_s=0.0, failures_to_alert=3)
    events = _run_assistant_loop(
        monkeypatch,
        cfg,
        probes=[(False, "timeout"), (True, "ok"), (False, "timeout"), (True, "ok")],
        handle_recovered=True,
    )
    assert events == []


def test_assistant_remediate_confirm_reprobe_averts_restart(monkeypatch):
    """The confirm re-probe gates the SIGKILL: a failure that clears within
    the confirm window returns recovered with NO restart issued."""
    from pypoe.lab.config import AssistantSection, LabConfig

    cfg = LabConfig(
        assistant=AssistantSection(
            service_name="svc", env_root="/nowhere",
            restart_wait_s=0.0, confirm_wait_s=0.01,
        )
    )
    rev = {"n": 0}
    restarts: list[str] = []

    async def fake_health(lab, url):
        rev["n"] += 1
        # Baseline probe fails (the stall); the confirm re-probe succeeds.
        if rev["n"] >= 2:
            return True, "ok"
        return False, "timeout"

    monkeypatch.setattr(alert_routes, "_assistant_health_ok", fake_health)
    monkeypatch.setattr(alert_routes, "_service_active", lambda s: True)
    monkeypatch.setattr(
        alert_routes, "_restart_service",
        lambda s: restarts.append(s) or (True, "bounced"),
    )

    async def run():
        return await alert_routes._assistant_remediate(cfg, None)  # type: ignore[arg-type]

    recovered, steps = asyncio.run(run())
    assert recovered is True
    assert restarts == []
    assert any("confirm re-probe" in s[0] and s[1] for s in steps)


def test_dashboard_remediate_confirm_reprobe_averts_restart(monkeypatch):
    """Dashboard twin of the confirm gate."""
    cfg = _dash_cfg(service_name="svc", restart_wait_s=0.0, confirm_wait_s=0.01)
    rev = {"n": 0}
    restarts: list[str] = []

    async def fake_ok(lab, cfg):
        rev["n"] += 1
        if rev["n"] >= 2:
            return True, "ok"
        return False, "HTTP 500"

    monkeypatch.setattr(alert_routes, "_dashboard_ok", fake_ok)
    monkeypatch.setattr(alert_routes, "_service_active", lambda s: True)
    monkeypatch.setattr(
        alert_routes, "_restart_service",
        lambda s: restarts.append(s) or (True, "bounced"),
    )

    async def run():
        return await alert_routes._dashboard_remediate(cfg, None)  # type: ignore[arg-type]

    recovered, steps = asyncio.run(run())
    assert recovered is True
    assert restarts == []
    assert any("confirm re-probe" in s[0] and s[1] for s in steps)


# ---------------------------------------------------------------------------
# _post_slack renders mrkdwn, not CommonMark
# ---------------------------------------------------------------------------
# Every other test in this file monkeypatches _post_slack, so nothing else
# exercises what it actually hands Slack. These two do.


class _CapturingSlackClient:
    """Stands in for slack_sdk's AsyncWebClient, recording the payload."""

    captured: dict = {}

    def __init__(self, token=None):
        pass

    async def chat_postMessage(self, **kwargs):
        _CapturingSlackClient.captured = kwargs
        return {"ts": "1.0"}


@pytest.fixture()
def capture_slack(monkeypatch):
    import slack_sdk.web.async_client as async_client

    monkeypatch.setattr(async_client, "AsyncWebClient", _CapturingSlackClient)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    _CapturingSlackClient.captured = {}
    return _CapturingSlackClient


@pytest.mark.asyncio
async def test_post_slack_converts_investigator_markdown(capture_slack):
    """A model's markdown report reaches Slack as mrkdwn.

    This is the reported bug: `**Root cause:**` rendered as literal
    asterisks because Slack's text field is mrkdwn, where bold is a single
    asterisk.
    """
    await alert_routes._post_slack(
        "#lab-alerts", "**Root cause:** the OT-2 lost Wi-Fi.\n- [runbook](http://x/y)"
    )
    assert capture_slack.captured["text"] == (
        "*Root cause:* the OT-2 lost Wi-Fi.\n\u2022  <http://x/y|runbook>"
    )


@pytest.mark.asyncio
async def test_post_slack_leaves_hand_written_mrkdwn_alone(capture_slack):
    """PyPoe's own recovery lines are already mrkdwn and must not change."""
    line = ":white_check_mark: *SDL Assistant* recovered."
    await alert_routes._post_slack("#lab-alerts", line)
    assert capture_slack.captured["text"] == line
