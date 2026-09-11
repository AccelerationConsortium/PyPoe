"""``POST /alerts/kuma`` webhook handler.

Mounted onto PyPoe's existing FastAPI app (``interfaces/web/app.py``).
Receives Uptime Kuma's default JSON payload, posts an instant
"Investigating…" message into Slack, and kicks off a background
``codex exec`` invocation against the lab MCP server. The investigation
result is appended as a threaded reply.

Concurrency is bounded by a process-wide semaphore so a flood of alerts
doesn't fork an unbounded number of ``codex`` subprocesses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    from fastapi import APIRouter, FastAPI, HTTPException, Request, status
    from pydantic import BaseModel
    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - web-ui extra not installed
    APIRouter = None  # type: ignore[assignment]
    FastAPI = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[misc,assignment]
    status = None  # type: ignore[assignment]
    _FASTAPI_AVAILABLE = False

from ..core.mrkdwn import to_mrkdwn
from .config import load_config
from .http_client import LabClient

logger = logging.getLogger(__name__)

_MAX_INVESTIGATOR_OUTPUT_CHARS = 4000
_DEFAULT_MAX_CONCURRENT = 2

#: Investigator mandate, injected as Codex developer instructions.
SYSTEM_PROMPT = """You are the AC Organic Self-driving Lab's automated incident \
investigator. An alert has fired; investigate it using the read-only \
`pypoe-lab` MCP server and report a concise root-cause summary for a Slack \
thread.

You are READ-ONLY. You cannot actuate hardware and must never propose calling \
`/control/*` endpoints directly. If recovery needs a control action, recommend \
it in plain English for a human or a `lab-skills` workflow to carry out.

Ground every conclusion in evidence you actually read via the MCP tools. If the \
data does not support a conclusion, say so plainly rather than speculate.

Keep the final Slack reply concise, with a short labelled paragraph for each
consulted model and a final Lead investigator paragraph. Each model's section,
including your own, must be 100 words or fewer; this is a ceiling, not a target.
Summarise each model's diagnosis, supporting evidence, and recommended action.
In your conclusion, resolve meaningful disagreement and state the next action.
Say "unconfirmed" when the evidence is insufficient. If a consultation fails,
use a single short line under that model's name; never invent its opinion.
If no models were consulted, provide only your conclusion in 100 words or fewer.
Avoid tool-call narration, raw logs, repeated alert details, and repeated
arguments. Keep detailed findings in the journal observation."""


def _investigator_runtime_dir() -> Path:
    """Scratch cwd outside the repository, avoiding project instructions.

    Override with ``PYPOE_INVESTIGATOR_RUNTIME_DIR``.
    """

    d = Path(
        os.environ.get(
            "PYPOE_INVESTIGATOR_RUNTIME_DIR",
            str(Path.home() / ".pypoe" / "investigator"),
        )
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def _investigator_cwd() -> str:
    return os.environ.get("PYPOE_INVESTIGATOR_CWD") or str(_investigator_runtime_dir())


def _codex_binary() -> Optional[str]:
    """Resolve the ``codex`` CLI, tolerating a minimal systemd PATH.

    PyPoe's web service runs under systemd with a PATH that usually excludes
    ``~/.local/bin``, so ``shutil.which`` alone often misses a per-user install.
    Honour an explicit override, then fall back to well-known install paths.
    """

    override = os.environ.get("PYPOE_CODEX_BIN")
    if override:
        return override
    found = shutil.which("codex")
    if found:
        return found
    candidates = [
        Path.home() / ".local" / "bin" / "codex",
        Path("/usr/local/bin/codex"),
        Path("/opt/codex/bin/codex"),
        Path("/home/sdl2/.local/bin/codex"),
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


def _pypoe_binary() -> str:
    """Resolve the ``pypoe`` console script that launches the lab MCP server.

    Prefers ``PYPOE_BIN``, then PATH, then the script sitting next to the
    running interpreter (the common editable-install layout). Falls back to the
    bare name so a mis-resolved env still produces a legible spawn error.
    """

    override = os.environ.get("PYPOE_BIN")
    if override:
        return override
    found = shutil.which("pypoe")
    if found:
        return found
    sibling = Path(sys.executable).with_name("pypoe")
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return "pypoe"


def _codex_mcp_args(cfg) -> list[str]:
    """Configure only the lab server; pass secrets through env, never argv."""
    server = {
        "command": _pypoe_binary(),
        "args": ["lab-mcp"],
        "env_vars": [
            "POE_API_KEY", "OPENROUTER_API_KEY", "LAB_MCP_AGENT_SOURCE",
            "LAB_MCP_HTTP_TIMEOUT", "LAB_CONSULT_ENABLED", "LAB_CONSULT_MODELS",
            "PYPOE_LAB_CONFIG",
        ],
        "required": True,
        # Equivalent to the former Claude --allowedTools lab-server glob.
        "default_tools_approval_mode": "approve",
    }
    args = []
    for key, value in server.items():
        args.extend(["-c", f"mcp_servers.pypoe-lab.{key}={json.dumps(value)}"])
    args.extend(["-c", f"mcp_servers.pypoe-lab.env.LAB_API_URL={json.dumps(cfg.api_url)}"])
    return args


_INVESTIGATION_PROMPT_HEAD = """\
An Uptime Kuma alert just fired for monitor `{monitor}` (msg: {msg}).

You have a `pypoe-lab` MCP server registered. Use it to investigate.

Steps:

1. Call `aggregator_health()` and `list_equipment()`. If the aggregator
   itself is down, say so and stop.
2. For each device whose state is not `ready`/`idle`/`running`/`dry_run`,
   call `get_equipment_status()`, `recent_events(device_id)`, and
   `recent_observations(device_id)`. Inspect `status.equipment_status`,
   `status.message`, `status.last_error`, and `details.claimed_by`. If a
   device has prior observations, treat a repeat as a RECURRENCE and build on
   the earlier root cause instead of re-deriving it from scratch.
"""

_DEVICE_PROMPT_HEAD = """\
A lab device alert just fired: `{device_id}` reported `{event}`
(message: {msg}).{platform_line}{last_error_line}{devices_line}

You have a `pypoe-lab` MCP server registered. Use it to investigate.

Steps:

1. Call `get_equipment_status("{device_id}")`,
   `recent_events("{device_id}")`, and `recent_observations("{device_id}")`.
   Inspect `status.equipment_status`, `status.message`, `status.last_error`,
   and `details.claimed_by`. If `recent_observations` shows this device was
   investigated before, decide whether this is a NEW issue or a RECURRENCE:
   for a recurrence, build on the prior root cause (cite its date and note the
   occurrence count) rather than starting cold.
2. Call `device_uptime("{device_id}")` and `aggregator_health()` for
   context. If other devices are listed as affected above, check each
   of them briefly with `get_equipment_status()` — simultaneous failures
   usually mean a shared cause (network, gateway, power).
"""

_CONSULT_BLOCK = """\
3. Consult each Poe model below for an independent second opinion.
   For EACH model, call:
       consult_poe(model="<model>", question="<your question>", context="<lab state you gathered>")
   The `context` you pass MUST include the relevant facts you read in
   step 2 — the consulted model has no MCP access of its own, so the
   string you supply is the ONLY information it sees.

   Ask each model to answer in 100 words or fewer, covering its diagnosis,
   key evidence, and next action.

   Models to consult (one call per model):
{models_block}

   If a `consult_poe` call returns `returncode != 0` (Poe unreachable,
   bad bot, missing API key, etc.), record the failure in the journal
   and continue. Include a single short failure line under that model in Slack.
"""

_NO_CONSULT_BLOCK = """\
3. Skip Poe consultation (disabled in slack.yaml). If a human judgment
   call is needed, call `ask_human(...)`.
"""

_INVESTIGATION_PROMPT_TAIL_CONSULT = """\
4. Per affected device, call `append_observation(device_id, summary,
   severity)` to journal your finding. Lead `summary` with a stable one-line
   root-cause headline so a future `recent_observations` read can match a
   recurrence to this incident.

You CANNOT perform control actions through this server. Do not propose
calling `/control/*` directly. If recovery requires a control action,
recommend it in plain English so a human or workflow can execute it via
`lab-skills`.

End with a labelled summary for each consulted model, then your synthesised
Lead investigator conclusion. Each section must be 100 words or fewer.
Include key evidence and the next action; journal the details.
"""

_INVESTIGATION_PROMPT_TAIL_SOLO = """\
4. Per affected device, call `append_observation(device_id, summary,
   severity)` to journal your finding. Lead `summary` with a stable one-line
   root-cause headline so a future `recent_observations` read can match a
   recurrence to this incident.

You CANNOT perform control actions through this server. Do not propose
calling `/control/*` directly. If recovery requires a control action,
recommend it in plain English so a human or workflow can execute it via
`lab-skills`.

End with your Lead investigator conclusion in 100 words or fewer, covering
the likely cause, key evidence, and next action. Journal the details.
"""


def _build_investigation_prompt(
    monitor: str, msg: str, consult_models: tuple[str, ...]
) -> str:
    """Compose the investigation prompt with the configured model list.

    If ``consult_models`` is empty, switch to the solo prompt (no Poe
    consultation step, no synthesis instructions).
    """
    head = _INVESTIGATION_PROMPT_HEAD.format(monitor=monitor, msg=msg)
    return _finish_prompt(head, consult_models)


def _build_device_prompt(
    device_id: str,
    event: str,
    msg: str,
    consult_models: tuple[str, ...],
    *,
    last_error: Optional[dict] = None,
    devices: Optional[list[str]] = None,
    platform: Optional[str] = None,
    siblings: Optional[list[str]] = None,
) -> str:
    """Compose a device-focused investigation prompt (POST /alerts/device)."""
    last_error_line = (
        f"\nDevice-reported last_error: {last_error}" if last_error else ""
    )
    devices_line = (
        f"\nOther devices affected in the same sweep: {', '.join(devices)}"
        if devices
        else ""
    )
    # Platform context sharpens shared-cause reasoning: a whole-bench failure
    # (lost PC, gateway, power) tends to take out the co-located devices too.
    if platform:
        colocated = (
            f" (co-located devices: {', '.join(siblings)})" if siblings else ""
        )
        platform_line = f"\nPlatform: {platform}{colocated}."
    else:
        platform_line = ""
    head = _DEVICE_PROMPT_HEAD.format(
        device_id=device_id,
        event=event,
        msg=msg,
        platform_line=platform_line,
        last_error_line=last_error_line,
        devices_line=devices_line,
    )
    return _finish_prompt(head, consult_models)


def _finish_prompt(head: str, consult_models: tuple[str, ...]) -> str:
    if consult_models:
        models_block = "\n".join(f"     - {m}" for m in consult_models)
        return head + _CONSULT_BLOCK.format(models_block=models_block) + _INVESTIGATION_PROMPT_TAIL_CONSULT
    return head + _NO_CONSULT_BLOCK + _INVESTIGATION_PROMPT_TAIL_SOLO


class KumaHeartbeat(BaseModel):  # type: ignore[misc]
    status: int = 0
    msg: Optional[str] = None
    time: Optional[str] = None


class KumaMonitor(BaseModel):  # type: ignore[misc]
    name: Optional[str] = None
    type: Optional[str] = None


class KumaAlert(BaseModel):  # type: ignore[misc]
    heartbeat: Optional[KumaHeartbeat] = None
    monitor: Optional[KumaMonitor] = None
    msg: Optional[str] = None


#: Device-alert events the aggregator-side notifier may send. ``recovered``
#: posts a one-liner; everything else triggers an investigation.
DEVICE_ALERT_EVENTS = ("unreachable", "error", "e_stop", "degraded", "recovered")


class DeviceAlert(BaseModel):  # type: ignore[misc]
    """Payload for ``POST /alerts/device`` (sent by the lab aggregator's
    alert notifier, not by Uptime Kuma)."""

    device_id: str
    event: str  # one of DEVICE_ALERT_EVENTS
    state: Optional[str] = None
    message: Optional[str] = None
    last_error: Optional[dict] = None
    #: For storm-collapsed alerts: the other device ids that tripped in
    #: the same sweep (device_id carries the first one).
    devices: Optional[list[str]] = None


def register_alert_routes(
    app: "FastAPI",
    client: Optional[LabClient] = None,
    *,
    max_concurrent: Optional[int] = None,
    slack_channel: Optional[str] = None,
) -> "APIRouter":
    """Mount ``POST /alerts/kuma`` onto a FastAPI app."""
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for register_alert_routes — install the "
            "web-ui extra: pip install -e '.[web-ui]'"
        )

    lab = client or LabClient()
    cfg = load_config()
    concurrency = (
        max_concurrent
        if max_concurrent is not None
        else (cfg.alerts.max_concurrent_investigations or _DEFAULT_MAX_CONCURRENT)
    )
    semaphore = asyncio.Semaphore(concurrency)
    channel = slack_channel or cfg.slack.alert_channel

    router = APIRouter(prefix="/alerts", tags=["lab-alerts"])

    @router.post("/kuma", status_code=202)
    async def kuma_webhook(payload: KumaAlert, request: Request) -> dict:
        monitor_name = (payload.monitor.name if payload.monitor else None) or "unknown"
        msg = payload.msg or (payload.heartbeat.msg if payload.heartbeat else "") or ""
        is_down = payload.heartbeat is not None and payload.heartbeat.status == 0
        # Best-effort platform label. Kuma often watches *services* (aggregator,
        # gateways) that aren't in any platform section — those stay unlabelled.
        platform, _siblings = await _resolve_platform(lab, monitor_name)
        label = _platform_label(platform, monitor_name)

        if not is_down:
            # Recovery — post the recovery line, no investigation.
            try:
                await _post_slack(
                    channel,
                    f":white_check_mark: *{label}* recovered — {msg}".strip(),
                )
            except Exception as exc:
                logger.error("Failed to post Slack recovery alert: %s", exc)
            return {"action": "recovered", "monitor": monitor_name}

        # Down — post the "investigating" line and kick off a background task.
        try:
            thread_ts = await _post_slack(
                channel,
                f":rotating_light: *{label}* DOWN — {msg} "
                f":mag: Investigating…",
            )
        except Exception as exc:
            logger.error("Failed to post Slack alert: %s", exc)
            thread_ts = None

        # Investigation runs in the background, bounded by the semaphore.
        # Consult-model list is captured per request (so a slack.yaml edit
        # takes effect on the next alert without a process restart, if the
        # config singleton has been reload_config()-ed).
        consult_models = cfg.consult.models if cfg.consult.enabled else ()
        prompt = _build_investigation_prompt(monitor_name, msg, consult_models)
        asyncio.create_task(
            _investigate(
                prompt=prompt,
                channel=channel,
                thread_ts=thread_ts,
                semaphore=semaphore,
            )
        )
        return {
            "action": "investigating",
            "monitor": monitor_name,
            "thread_ts": thread_ts,
        }

    @router.post("/device", status_code=202)
    async def device_alert(payload: DeviceAlert) -> dict:
        if payload.event not in DEVICE_ALERT_EVENTS:
            raise HTTPException(
                status_code=422,
                detail=f"event must be one of {DEVICE_ALERT_EVENTS}",
            )
        msg = payload.message or ""
        # Best-effort: label the alert with its platform, and hand the
        # investigator the co-located devices for shared-cause reasoning.
        platform, siblings = await _resolve_platform(lab, payload.device_id)
        label = _platform_label(platform, payload.device_id)

        if payload.event == "recovered":
            try:
                await _post_slack(
                    channel,
                    f":white_check_mark: *{label}* recovered — {msg}".strip(),
                )
            except Exception as exc:
                logger.error("Failed to post device recovery alert: %s", exc)
            return {"action": "recovered", "device": payload.device_id}

        others = f" (+{len(payload.devices)} more)" if payload.devices else ""
        try:
            thread_ts = await _post_slack(
                channel,
                f":rotating_light: *{label}*{others} "
                f"{payload.event.upper()} — {msg} :mag: Investigating…",
            )
        except Exception as exc:
            logger.error("Failed to post device alert: %s", exc)
            thread_ts = None

        consult_models = cfg.consult.models if cfg.consult.enabled else ()
        prompt = _build_device_prompt(
            payload.device_id,
            payload.event,
            msg,
            consult_models,
            last_error=payload.last_error,
            devices=payload.devices,
            platform=platform,
            siblings=siblings,
        )
        asyncio.create_task(
            _investigate(
                prompt=prompt,
                channel=channel,
                thread_ts=thread_ts,
                semaphore=semaphore,
            )
        )
        return {
            "action": "investigating",
            "device": payload.device_id,
            "thread_ts": thread_ts,
        }

    app.include_router(router)
    return router


# ---------------------------------------------------------------------------
# Background investigation
# ---------------------------------------------------------------------------


async def _investigate(
    *,
    prompt: str,
    channel: str,
    thread_ts: Optional[str],
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        try:
            output = await _run_codex(prompt)
        except Exception as exc:
            output = f":x: Investigation failed to start: {exc}"
        if len(output) > _MAX_INVESTIGATOR_OUTPUT_CHARS:
            output = output[: _MAX_INVESTIGATOR_OUTPUT_CHARS - 100] + "\n…(truncated)"
        try:
            await _post_slack(channel, output, thread_ts=thread_ts)
        except Exception as exc:
            logger.error("Failed to post investigation reply: %s", exc)


async def _run_codex(prompt: str) -> str:
    """Run Codex with pinned reasoning, isolated lab MCP tools and a timeout."""
    binary = _codex_binary()
    if binary is None:
        return ":x: `codex` CLI not on PATH. Install Codex or set `PYPOE_CODEX_BIN`."

    cfg = load_config()
    timeout_s = cfg.alerts.investigation_timeout_s
    args = [
        binary, "exec", "--ignore-user-config", "--ignore-rules",
        "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
        "--model", cfg.alerts.investigation_model,
        "-c", f"model_reasoning_effort={json.dumps(cfg.alerts.investigation_reasoning_effort)}",
        "-c", "approval_policy=\"never\"",
        "-c", f"developer_instructions={json.dumps(SYSTEM_PROMPT)}",
        "-c", "features.shell_tool=false",
        "-c", "features.unified_exec=false",
        "-c", "features.apps=false",
        "-c", "web_search=\"disabled\"",
        "-c", "project_doc_max_bytes=0",
        "-c", f"sqlite_home={json.dumps(str(_investigator_runtime_dir()))}",
        "-c", f"log_dir={json.dumps(str(_investigator_runtime_dir()))}",
        *_codex_mcp_args(cfg),
        "--color", "never", prompt,
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=_investigator_cwd(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return f":x: could not spawn `{binary}` — check the install / `PYPOE_CODEX_BIN`."

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        return (
            f":x: investigation exceeded {timeout_s:.0f}s timeout and was killed "
            "before producing a summary."
        )

    if proc.returncode != 0:
        return (
            f":x: `codex` exited {proc.returncode}\n"
            f"```\n{stderr.decode(errors='replace').strip()[:1500]}\n```"
        )
    return stdout.decode(errors="replace").strip()


# ---------------------------------------------------------------------------
# Platform resolution (device_id -> platform label + co-located siblings)
# ---------------------------------------------------------------------------


def _platform_label(platform: Optional[str], device_id: str) -> str:
    """Slack headline label: ``"HTE Platform · ot2_hte"`` when the platform is
    known, else just the device id."""
    return f"{platform} · {device_id}" if platform else device_id


async def _resolve_platform(
    lab: LabClient, device_id: str
) -> tuple[Optional[str], list[str]]:
    """Best-effort ``(platform_title, co-located device ids)`` for a device.

    Reads the aggregator's platforms view (``GET /api/platforms``) and finds the
    section whose equipment list contains ``device_id``. Returns ``(None, [])``
    when the device isn't in any section (e.g. a Kuma *service* monitor) or the
    lookup fails — never raises, so a platforms outage never blocks an alert.
    """
    try:
        data = await lab.platforms()
    except Exception:
        return None, []
    for section in (data.get("sections") or []):
        equipment = section.get("equipment") or []
        if device_id in equipment:
            title = section.get("title") or section.get("id")
            siblings = [e for e in equipment if e != device_id]
            return title, siblings
    return None, []


# ---------------------------------------------------------------------------
# Slack posting (kept tiny so tests can monkeypatch it)
# ---------------------------------------------------------------------------


async def _post_slack(
    channel: str, text: str, thread_ts: Optional[str] = None
) -> Optional[str]:
    try:
        from slack_sdk.web.async_client import AsyncWebClient
    except ImportError as exc:
        logger.warning("slack_sdk missing: %s", exc)
        return None

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        logger.warning("SLACK_BOT_TOKEN unset — not posting to %s", channel)
        return None

    slack = AsyncWebClient(token=token)
    # Slack renders `text` as mrkdwn, not CommonMark. Every alert line
    # funnels through here — our own hand-written mrkdwn (which the
    # converter leaves alone) and the investigator's markdown report alike
    # — so this is the one place the dialect gap has to be closed.
    kwargs = {"channel": channel, "text": to_mrkdwn(text)}
    if thread_ts:
        kwargs["thread_ts"] = thread_ts
    resp = await slack.chat_postMessage(**kwargs)
    return resp.get("ts")


# ---------------------------------------------------------------------------
# SDL Assistant self-healing monitor
# ---------------------------------------------------------------------------
# Watches the assistant's /api/assistant/health endpoint. On a DOWN transition
# it posts a Slack alert, attempts a bounded set of common fixes (verify /
# restart the backing API service, check the OpenRouter key is present),
# re-probes after each, and posts a threaded report of what succeeded vs
# failed. Recovery posts a :white_check_mark: line. Runs either as an
# in-process periodic task (AssistantSection.monitor_enabled) or on demand via
# POST /alerts/assistant.


def _service_mainpid(service: str) -> Optional[str]:
    """Best-effort MainPID of a systemd unit, without sudo.

    ``systemctl show -p MainPID --value`` needs no privileges to *read*. The
    service is then relaunched via ``kill`` (see ``_restart_service``); this
    only works because the unit runs as the same user with Restart=on-failure.
    """
    try:
        out = subprocess.run(
            ["systemctl", "show", service, "-p", "MainPID", "--value"],
            capture_output=True, text=True, timeout=10,
        )
        pid = out.stdout.strip()
        return pid if pid.isdigit() else None
    except Exception:  # pragma: no cover - best-effort
        return None


def _service_active(service: str) -> bool:
    """Whether a systemd unit is currently active (is-active exit 0)."""
    try:
        return subprocess.run(
            ["systemctl", "is-active", service],
            capture_output=True, text=True, timeout=10,
        ).returncode == 0
    except Exception:  # pragma: no cover - best-effort
        return False


def _restart_service(service: str) -> tuple[bool, str]:
    """Relaunch a user-owned Restart=on-failure unit without sudo.

    SIGKILL the unit's MainPID; systemd sees the abnormal exit and relaunches it
    within RestartSec (default ~100ms-5s). Falls back to ``systemctl restart``
    (which may require policy) if no PID is resolvable.
    """
    pid = _service_mainpid(service)
    if pid:
        try:
            subprocess.run(
                ["kill", "-KILL", pid], capture_output=True, text=True, timeout=10,
            )
            return True, f"killed PID {pid}; Restart=on-failure relaunches {service}"
        except Exception as exc:  # pragma: no cover
            return False, f"kill failed: {exc}"
    try:
        subprocess.run(
            ["systemctl", "restart", service],
            capture_output=True, text=True, timeout=30,
        )
        return True, f"systemctl restart {service}"
    except Exception as exc:  # pragma: no cover
        return False, f"restart failed: {exc}"


def _assistant_key_present(env_root: str) -> tuple[bool, str]:
    """Confirm ASSISTANT_OPENAI_API_KEY is set (non-empty) in the repo .env.

    This is the classic silent-failure cause: the assistant health endpoint
    reports ``configured:false`` when the key is missing. We only check
    presence — never read the value back into the transcript.
    """
    try:
        p = Path(env_root) / ".env"
        if not p.is_file():
            return False, f"{p} not found"
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("ASSISTANT_OPENAI_API_KEY="):
                val = line.split("=", 1)[1].strip()
                if val and not val.startswith('"') and not val.startswith("'"):
                    return True, "key present"
        return False, "ASSISTANT_OPENAI_API_KEY missing/unset in .env"
    except Exception as exc:  # pragma: no cover
        return False, f"could not read .env: {exc}"


async def _assistant_health_ok(
    lab: "LabClient", url: str
) -> tuple[bool, str]:
    """Probe the assistant health endpoint; return (healthy, human detail).

    Treats as healthy when the request succeeds and the JSON payload reports
    ``configured: true`` (the same signal that gates the assistant bubble).
    Any transport error, non-200, or ``configured:false`` is a DOWN.
    """
    detail = f"GET {url}"
    try:
        resp = await lab._client.get(url)
    except Exception as exc:  # httpx errors
        return False, f"{detail} -> unreachable ({type(exc).__name__})"
    if resp.status_code != 200:
        return False, f"{detail} -> HTTP {resp.status_code}"
    try:
        data = resp.json()
    except Exception:
        return False, f"{detail} -> non-JSON response"
    if not isinstance(data, dict):
        return False, f"{detail} -> unexpected payload shape"
    configured = data.get("configured")
    if configured is not True:
        return False, (
            f"{detail} -> configured=false "
            f"(backend={data.get('backend')!r}); likely missing OpenRouter key"
        )
    return True, (
        f"{detail} -> ok (backend={data.get('backend')!r}, "
        f"model={data.get('model')!r})"
    )


async def _assistant_remediate(
    cfg, lab: "LabClient"
) -> tuple[bool, list[tuple[str, bool, str]]]:
    """Attempt a bounded set of common fixes; return (healthy_now, steps).

    Each step is ``(label, succeeded, detail)``. Steps:
      1. Re-probe (baseline) — if the service is down but health recovers on a
         plain retry (transient), we're done.
      2. Verify the backing API service is active; if not, try to start it.
      3. Restart the service (a wedged-but-running process needs a bounce).
      4. Verify the OpenRouter key is present (report-only fix; we cannot mint a
         secret). Re-probe after each mutating step.
    """
    from .config import load_config
    cfg = cfg or load_config()
    asst = cfg.assistant
    steps: list[tuple[str, bool, str]] = []

    ok, det = await _assistant_health_ok(lab, asst.health_url)
    if ok:
        return True, [("baseline re-probe", True, det)]

    # 1. If the backing service is down, try to start it.
    if not _service_active(asst.service_name):
        st_ok, st_det = _restart_service(asst.service_name)
        steps.append((f"start {asst.service_name}", st_ok, st_det))
        await asyncio.sleep(asst.restart_wait_s)
        ok, det = await _assistant_health_ok(lab, asst.health_url)
        if ok:
            return True, steps
    else:
        steps.append((f"service {asst.service_name} active", True, "active"))

    # 2. Bounce the service (wedged process still answers on the port) — but
    #    only after confirming the failure is the service's own. Under a
    #    machine-wide stall (memory/IO reclaim freezing every process on the
    #    host, as on 2026-08-30) every probe times out while the service is
    #    perfectly healthy, and a SIGKILL here *causes* the outage it thinks
    #    it is fixing. Wait out the confirm window and re-probe first.
    if asst.confirm_wait_s > 0:
        await asyncio.sleep(asst.confirm_wait_s)
        ok, det = await _assistant_health_ok(lab, asst.health_url)
        if ok:
            return True, steps + [
                ("confirm re-probe", True, f"cleared on its own — no restart: {det}")
            ]
        steps.append(("confirm re-probe", False, det))
    b_ok, b_det = _restart_service(asst.service_name)
    steps.append((f"restart {asst.service_name}", b_ok, b_det))
    await asyncio.sleep(asst.restart_wait_s)
    ok, det = await _assistant_health_ok(lab, asst.health_url)
    if ok:
        return True, steps

    # 3. Verify the key (report-only).
    k_ok, k_det = _assistant_key_present(asst.env_root)
    steps.append(("ASSISTANT_OPENAI_API_KEY present", k_ok, k_det))

    return False, steps + [("final re-probe", False, det)]


async def _report_remediation(
    channel: str, thread_ts: Optional[str], recovered: bool, steps
) -> None:
    lines = []
    for label, ok, det in steps:
        mark = ":white_check_mark:" if ok else ":x:"
        lines.append(f"{mark} {label} — {det}")
    head = (
        f"*SDL Assistant* :green_circle: recovered after self-heal:"
        if recovered
        else f"*SDL Assistant* :red_circle: STILL DOWN after self-heal:"
    )
    text = "\n".join([head, *lines])
    try:
        await _post_slack(channel, text, thread_ts=thread_ts)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to post SDL assistant remediation report: %s", exc)


async def _assistant_handle_down(
    channel: str, lab: "LabClient", cfg, det: str
) -> bool:
    """Post the DOWN alert for an observed failure, remediate, report.

    Returns True when remediation left the assistant healthy again (the
    threaded report then already ends on the :green_circle: line). The caller
    supplies ``det`` from the probe that actually failed, so the alert always
    describes the observed failure rather than a fresh re-probe.
    """
    try:
        thread_ts = await _post_slack(
            channel,
            f":rotating_light: *SDL Assistant* DOWN — {det} :mag: trying fixes…",
        )
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to post SDL assistant alert: %s", exc)
        thread_ts = None

    recovered, steps = await _assistant_remediate(cfg, lab)
    await _report_remediation(channel, thread_ts, recovered, steps)
    return recovered


async def _assistant_alert_once(
    channel: str, lab: "LabClient", cfg
) -> Optional[bool]:
    """On-demand down-handling: probe, and on failure alert -> remediate.

    Returns None when the assistant was healthy (nothing posted — the manual
    POST doubles as a liveness check), else whether remediation recovered it.
    """
    asst = cfg.assistant
    ok, det = await _assistant_health_ok(lab, asst.health_url)
    if ok:
        return None
    return await _assistant_handle_down(channel, lab, cfg, det)


async def _assistant_recover(channel: str) -> None:
    try:
        await _post_slack(channel, ":white_check_mark: *SDL Assistant* recovered.")
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to post SDL assistant recovery: %s", exc)


async def _assistant_monitor_loop(
    channel: str, lab: "LabClient", cfg, semaphore: "asyncio.Semaphore"
) -> None:
    """Periodic probe loop. Alerts on a DOWN transition, recovers on UP.

    ``was_down`` means exactly "a DOWN alert stands unanswered in the
    channel". It used to be latched *before* the (re-probing) alert call, so
    a transient failure that cleared in between posted no outage line yet
    still posted ':white_check_mark: recovered.' on the next healthy tick —
    the orphan-recovery Slack noise of 2026-08-30. The transition branch now
    alerts on the failing probe it already observed and arms the latch only
    while the outage is actually unresolved.
    """
    asst = cfg.assistant
    was_down = False
    consecutive_failures = 0
    while True:
        await asyncio.sleep(asst.probe_interval_s)
        ok, det = await _assistant_health_ok(lab, asst.health_url)
        if ok:
            consecutive_failures = 0
            if was_down:
                was_down = False
                await _assistant_recover(channel)
            continue
        consecutive_failures += 1
        if consecutive_failures < asst.failures_to_alert:
            continue
        if was_down:
            # Already alerting earlier — re-run remediation quietly on each
            # tick (rate-limited by the semaphore) but don't re-fire the alert.
            try:
                async with semaphore:
                    recovered, steps = await _assistant_remediate(cfg, lab)
                    if recovered:
                        was_down = False
                        await _assistant_recover(channel)
                    else:
                        await _report_remediation(channel, None, False, steps)
            except Exception as exc:  # pragma: no cover
                logger.error("SDL assistant monitor remediation failed: %s", exc)
            continue
        recovered = False
        try:
            async with semaphore:
                recovered = await _assistant_handle_down(channel, lab, cfg, det)
        except Exception as exc:  # pragma: no cover
            logger.error("SDL assistant monitor alert failed: %s", exc)
        # When remediation already recovered it, the threaded report ended on
        # the :green_circle: line — leaving the latch armed would add a
        # second, orphan recovery on the next tick.
        was_down = not recovered
        if recovered:
            consecutive_failures = 0


def register_assistant_monitor(
    app: "FastAPI",
    client: Optional["LabClient"] = None,
    *,
    slack_channel: Optional[str] = None,
) -> None:
    """Mount ``POST /alerts/assistant`` and (if enabled) the probe loop.

    The background monitor is started on the app's ``startup`` (where a running
    event loop exists) and cancelled on ``shutdown`` — the route is registered
    here so callers keep the same simple API, but the loop only spins once the
    server is actually serving. Returns nothing.

    The route is idempotent: a manual POST when the assistant is healthy posts
    nothing and acts as a liveness check. Monitoring is a no-op unless
    ``assistant.monitor_enabled`` is true (config/env).
    """
    if not _FASTAPI_AVAILABLE:
        return
    from .config import load_config
    cfg = load_config()
    channel = slack_channel or cfg.slack.alert_channel
    lab = client or LabClient()

    @app.post("/alerts/assistant", status_code=202)
    async def assistant_alert() -> dict:
        # On-demand self-heal (e.g. an external scheduler or a manual poke).
        await _assistant_alert_once(channel, lab, cfg)
        return {"action": "assistant_selfheal"}

    if not cfg.assistant.monitor_enabled:
        return

    _state: dict[str, Optional["asyncio.Task"]] = {"task": None}

    @app.on_event("startup")
    async def _start_monitor() -> None:
        semaphore = asyncio.Semaphore(1)
        _state["task"] = asyncio.create_task(
            _assistant_monitor_loop(channel, lab, cfg, semaphore)
        )

    @app.on_event("shutdown")
    async def _stop_monitor() -> None:
        task = _state.get("task")
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            _state["task"] = None


# ---------------------------------------------------------------------------
# SDL Dashboard surface monitor
# ---------------------------------------------------------------------------
# Watches the dashboard's *data* routes (/api/openapi.json, /api/catalog,
# /api/equipment) rather than /api/health. Health only proves the server is
# awake; a stale or half-broken deploy can keep health green while the real
# UI-serving routes throw 500s (Jiaru's case: /api/openapi.json 500 with
# /api/health healthy). On a DOWN transition it posts a Slack alert, bounces
# the backing API service, re-probes, and reports what succeeded vs failed.


async def _dashboard_ok(
    lab: "LabClient", cfg
) -> tuple[bool, str]:
    """Probe every dashboard path; return (healthy, human detail).

    DOWN = any transport error, non-``expected_status``, or non-dict JSON body.
    """
    dash = cfg.dashboard
    results: list[str] = []
    for path in dash.paths:
        url = _dashboard_url(dash.base_url, path)
        try:
            resp = await lab._client.get(url)
        except Exception as exc:  # httpx errors
            results.append(f"{path} -> unreachable ({type(exc).__name__})")
            continue
        if resp.status_code != dash.expected_status:
            results.append(f"{path} -> HTTP {resp.status_code}")
            continue
        try:
            data = resp.json()
        except Exception:
            results.append(f"{path} -> non-JSON response")
            continue
        if dash.expected_status == 200 and not isinstance(data, dict):
            results.append(f"{path} -> unexpected payload shape")
            continue
        results.append(f"{path} -> HTTP {dash.expected_status} ok")
    ok = all("ok" in r for r in results)
    return ok, "; ".join(results)


def _dashboard_url(base: str, path: str) -> str:
    """Join a base URL and an absolute path, tolerating a trailing slash."""
    base = base.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}{path}"


async def _dashboard_remediate(
    cfg, lab: "LabClient"
) -> tuple[bool, list[tuple[str, bool, str]]]:
    """Attempt a bounded set of common fixes; return (healthy_now, steps).

    Steps:
      1. Re-probe (baseline) — a transient 500 may clear itself.
      2. If the backing service is inactive, try to start it.
      3. Bounce the service (a wedged-but-running process needs a restart).
    Re-probe after each step.
    """
    dash = cfg.dashboard
    steps: list[tuple[str, bool, str]] = []

    ok, det = await _dashboard_ok(lab, cfg)
    if ok:
        return True, [("baseline re-probe", True, det)]

    if not _service_active(dash.service_name):
        st_ok, st_det = _restart_service(dash.service_name)
        steps.append((f"start {dash.service_name}", st_ok, st_det))
        await asyncio.sleep(dash.restart_wait_s)
        ok, det = await _dashboard_ok(lab, cfg)
        if ok:
            return True, steps
    else:
        steps.append((f"service {dash.service_name} active", True, "active"))

    # Same SIGKILL gate as the assistant remediation: a machine-wide stall
    # fails these probes with the service perfectly healthy, and the bounce
    # would then cause the outage. Confirm before the kill.
    if dash.confirm_wait_s > 0:
        await asyncio.sleep(dash.confirm_wait_s)
        ok, det = await _dashboard_ok(lab, cfg)
        if ok:
            return True, steps + [
                ("confirm re-probe", True, f"cleared on its own — no restart: {det}")
            ]
        steps.append(("confirm re-probe", False, det))
    b_ok, b_det = _restart_service(dash.service_name)
    steps.append((f"restart {dash.service_name}", b_ok, b_det))
    await asyncio.sleep(dash.restart_wait_s)
    ok, det = await _dashboard_ok(lab, cfg)
    if ok:
        return True, steps

    return False, steps + [("final re-probe", False, det)]


async def _dashboard_report(
    channel: str, thread_ts: Optional[str], recovered: bool, steps
) -> None:
    lines = []
    for label, ok, det in steps:
        mark = ":white_check_mark:" if ok else ":x:"
        lines.append(f"{mark} {label} — {det}")
    head = (
        f"*SDL Dashboard* :green_circle: recovered after self-heal:"
        if recovered
        else f"*SDL Dashboard* :red_circle: STILL DOWN after self-heal:"
    )
    text = "\n".join([head, *lines])
    try:
        await _post_slack(channel, text, thread_ts=thread_ts)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to post SDL dashboard remediation report: %s", exc)


async def _dashboard_handle_down(
    channel: str, lab: "LabClient", cfg, det: str
) -> bool:
    """Post the DOWN alert for an observed failure, remediate, report.

    Mirrors ``_assistant_handle_down``: ``det`` comes from the probe that
    actually failed; returns True when remediation recovered the surface.
    """
    try:
        thread_ts = await _post_slack(
            channel,
            f":rotating_light: *SDL Dashboard* DOWN — {det} :mag: trying fixes…",
        )
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to post SDL dashboard alert: %s", exc)
        thread_ts = None

    recovered, steps = await _dashboard_remediate(cfg, lab)
    await _dashboard_report(channel, thread_ts, recovered, steps)
    return recovered


async def _dashboard_alert_once(
    channel: str, lab: "LabClient", cfg
) -> Optional[bool]:
    """On-demand down-handling: probe, and on failure alert -> remediate.

    Returns None when the dashboard was healthy (nothing posted), else
    whether remediation recovered it.
    """
    ok, det = await _dashboard_ok(lab, cfg)
    if ok:
        return None
    return await _dashboard_handle_down(channel, lab, cfg, det)


async def _dashboard_recover(channel: str) -> None:
    try:
        await _post_slack(channel, ":white_check_mark: *SDL Dashboard* recovered.")
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to post SDL dashboard recovery: %s", exc)


async def _dashboard_monitor_loop(
    channel: str, lab: "LabClient", cfg, semaphore: "asyncio.Semaphore"
) -> None:
    """Periodic probe loop. Alerts on a DOWN transition, recovers on UP.

    Same latch discipline as ``_assistant_monitor_loop``: ``was_down`` means
    "a DOWN alert stands unanswered", so a transient failure the remediation
    (or its baseline re-probe) already resolved never produces an orphan
    recovery line.
    """
    dash = cfg.dashboard
    was_down = False
    consecutive_failures = 0
    while True:
        await asyncio.sleep(dash.probe_interval_s)
        ok, det = await _dashboard_ok(lab, cfg)
        if ok:
            consecutive_failures = 0
            if was_down:
                was_down = False
                await _dashboard_recover(channel)
            continue
        consecutive_failures += 1
        if consecutive_failures < dash.failures_to_alert:
            continue
        if was_down:
            try:
                async with semaphore:
                    recovered, steps = await _dashboard_remediate(cfg, lab)
                    if recovered:
                        was_down = False
                        await _dashboard_recover(channel)
                    else:
                        await _dashboard_report(channel, None, False, steps)
            except Exception as exc:  # pragma: no cover
                logger.error("SDL dashboard monitor remediation failed: %s", exc)
            continue
        recovered = False
        try:
            async with semaphore:
                recovered = await _dashboard_handle_down(channel, lab, cfg, det)
        except Exception as exc:  # pragma: no cover
            logger.error("SDL dashboard monitor alert failed: %s", exc)
        was_down = not recovered
        if recovered:
            consecutive_failures = 0


def register_dashboard_monitor(
    app: "FastAPI",
    client: Optional["LabClient"] = None,
    *,
    slack_channel: Optional[str] = None,
) -> None:
    """Mount ``POST /alerts/dashboard`` and (if enabled) the probe loop.

    Mirrors ``register_assistant_monitor``: the route is always mounted (so a
    manual POST acts as a liveness/self-heal poke), but the background loop only
    spins when ``dashboard.monitor_enabled`` is true (config/env), and only on
    the app's ``startup`` where a running event loop exists.
    """
    if not _FASTAPI_AVAILABLE:
        return
    from .config import load_config
    cfg = load_config()
    channel = slack_channel or cfg.slack.alert_channel
    lab = client or LabClient()

    @app.post("/alerts/dashboard", status_code=202)
    async def dashboard_alert() -> dict:
        await _dashboard_alert_once(channel, lab, cfg)
        return {"action": "dashboard_selfheal"}

    if not cfg.dashboard.monitor_enabled:
        return

    _state: dict[str, Optional["asyncio.Task"]] = {"task": None}

    @app.on_event("startup")
    async def _start_dashboard_monitor() -> None:
        semaphore = asyncio.Semaphore(1)
        _state["task"] = asyncio.create_task(
            _dashboard_monitor_loop(channel, lab, cfg, semaphore)
        )

    @app.on_event("shutdown")
    async def _stop_dashboard_monitor() -> None:
        task = _state.get("task")
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            _state["task"] = None


