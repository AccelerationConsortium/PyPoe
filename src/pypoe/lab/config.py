"""Lab-integration config loader.

Loads runtime knobs for the ``pypoe.lab.*`` modules from a single YAML
file at ``src/pypoe/config/slack.yaml`` (gitignored; copy from
``slack.example.yaml`` next to it). Env vars take precedence so quick
overrides and CI don't require editing the YAML.

Why YAML instead of just ``.env``? The lab integration grew enough
knobs (api_url, alert_channel, command_prefix, mcp settings, alert
concurrency, consult model list) that a single typed config file is
easier to read and share than a flat env-var soup. ``.env`` keeps
secrets only (Slack bot token, signing secret, Poe API key).

Precedence (highest wins):
  1. Explicit kwargs in code (e.g. ``LabClient(base_url=...)``).
  2. Environment variables (``LAB_API_URL``, ``LAB_SLACK_CHANNEL``,
     ``LAB_SLACK_COMMAND_PREFIX``, ``LAB_ALERT_MAX_CONCURRENT``,
     ``LAB_INVESTIGATION_MODEL``, ``LAB_INVESTIGATION_TIMEOUT_S``,
     ``LAB_MCP_AGENT_SOURCE``, ``LAB_MCP_HTTP_TIMEOUT``).
  3. Values in the YAML config.
  4. Hardcoded defaults below.

The YAML file is **optional**. Existing deployments that only set
env vars keep working unchanged.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Resolution order for finding the YAML file.
DEFAULT_FILENAME = "slack.yaml"
EXAMPLE_FILENAME = "slack.example.yaml"


@dataclass(frozen=True)
class SlackSection:
    alert_channel: str = "#lab-alerts"
    command_prefix: str = "/lab-"


@dataclass(frozen=True)
class AlertsSection:
    max_concurrent_investigations: int = 2
    #: Lead investigator runs through local Codex authentication, not Poe.
    #: Env: ``LAB_INVESTIGATION_MODEL`` / ``LAB_INVESTIGATION_REASONING_EFFORT``.
    investigation_model: str = "gpt-5.6-luna"
    investigation_reasoning_effort: str = "max"
    #: Hard wallclock cap (seconds) on a single ``codex`` investigation
    #: subprocess. Generous by default because an investigation fans out to
    #: several MCP reads plus per-model ``consult_poe`` round-trips, but bounded
    #: so a hung CLI can never linger. Env: ``LAB_INVESTIGATION_TIMEOUT_S``.
    investigation_timeout_s: float = 300.0


@dataclass(frozen=True)
class McpSection:
    agent_source: str = "claude-agent"
    http_timeout_s: float = 10.0


@dataclass(frozen=True)
class AssistantSection:
    """Self-healing monitor for the SDL assistant (health endpoint + service).

    When ``monitor_enabled`` is True, a background task in the PyPoe web
    service probes the assistant's ``/api/assistant/health`` every
    ``probe_interval_s`` seconds. On a DOWN transition it posts a Slack alert,
    attempts a bounded set of common fixes (verify / restart the backing API
    service, check the OpenRouter key is present), re-probes after each, and
    reports what succeeded vs failed as a threaded reply; it posts a recovery
    line when health returns. See ``alert_routes`` (``AssistantMonitor``).

    The backing service runs as ``sdl2`` with ``Restart=on-failure``, so a
    ``kill`` of its MainPID is enough to relaunch it without sudo.

    Env overrides:
        LAB_ASSISTANT_MONITOR_ENABLED, LAB_ASSISTANT_HEALTH_URL,
        LAB_ASSISTANT_PROBE_INTERVAL_S, LAB_ASSISTANT_SERVICE,
        LAB_ASSISTANT_ENV_ROOT, LAB_ASSISTANT_RESTART_WAIT_S,
        LAB_ASSISTANT_FAILURES_TO_ALERT, LAB_ASSISTANT_CONFIRM_WAIT_S.
    """

    monitor_enabled: bool = False
    health_url: str = "http://127.0.0.1:8001/api/assistant/health"
    probe_interval_s: float = 60.0
    service_name: str = "ac-organic-lab-api.service"
    env_root: str = "/home/sdl2/caoyang/ac-organic-lab"
    restart_wait_s: float = 10.0
    #: consecutive failing probes before an alert fires (damp flapping).
    #: 3, not 1: a single probe lost to a machine-wide memory/IO stall (the
    #: 2026-08-30 gaia thrash — ~25 s freezes of every process) must not arm
    #: the monitor, let alone reach for the SIGKILL bounce.
    failures_to_alert: int = 3
    #: seconds to wait before the confirm re-probe that gates the SIGKILL
    #: bounce in remediation — long enough to ride out a machine-wide reclaim
    #: stall, which is not the service's fault. 0 disables the gate.
    confirm_wait_s: float = 20.0


@dataclass(frozen=True)
class DashboardSection:
    """Self-healing monitor for the SDL dashboard's non-liveness surfaces.

    ``GET /api/health`` only proves the server is awake — it does NOT exercise
    the routes that actually serve the UI (and that are the ones a stale or
    half-broken deploy breaks): the OpenAPI document, the skill catalog, and
    the equipment/status listings. Jiaru hit a real case: a single unresolved
    return annotation in the voice router made ``GET /api/openapi.json`` return
    500 while ``/api/health`` stayed green, so no monitor would have caught it.
    This monitor probes each configured path and treats any non-``expected_status``
    (or transport error / non-JSON body) as DOWN, then bounces the backing API
    service (SIGKILL relaunch — the unit runs as ``sdl2`` with
    ``Restart=on-failure``).

    Env overrides:
        LAB_DASHBOARD_MONITOR_ENABLED, LAB_DASHBOARD_BASE_URL,
        LAB_DASHBOARD_PATHS, LAB_DASHBOARD_PROBE_INTERVAL_S,
        LAB_DASHBOARD_SERVICE, LAB_DASHBOARD_RESTART_WAIT_S,
        LAB_DASHBOARD_EXPECTED_STATUS, LAB_DASHBOARD_FAILURES_TO_ALERT,
        LAB_DASHBOARD_CONFIRM_WAIT_S.
    """

    monitor_enabled: bool = False
    base_url: str = "http://127.0.0.1:8001"
    paths: tuple[str, ...] = (
        "/api/openapi.json",
        "/api/catalog",
        "/api/equipment",
    )
    probe_interval_s: float = 120.0
    service_name: str = "ac-organic-lab-api.service"
    restart_wait_s: float = 10.0
    expected_status: int = 200
    #: 2, not 1 — same machine-stall reasoning as AssistantSection (its
    #: probes are 120 s apart, so 2 already means ~4 min of failures).
    failures_to_alert: int = 2
    #: SIGKILL gate before the remediation bounce; see AssistantSection.
    confirm_wait_s: float = 20.0


@dataclass(frozen=True)
class ConsultSection:
    """Which Poe models the alert handler asks for a second opinion.

    When ``enabled`` is True (default), the investigation prompt
    *requires* the investigator to call ``consult_poe`` for each listed model
    and synthesise their responses into the Slack summary. When False
    (or ``models`` is empty), the lead investigates solo.

    These are reached through PyPoe's provider seam (distinct from the
    local-CLI investigator model above), so each name must appear in
    ``models.yaml::chat_models`` and is routed to whichever provider that
    entry declares. Env override: ``LAB_CONSULT_MODELS`` (comma-separated).

    A name that is not in the roster still resolves — to the default provider —
    and will fail at call time, so keep this list in step with the catalog. The
    previous default (``GPT-5.4``, ``GLM-5.2``) was Poe-routed and became
    unreachable when the Poe subscription lapsed.
    """

    enabled: bool = True
    models: tuple[str, ...] = ("z-ai/glm-5.2", "deepseek/deepseek-v4-flash-0731")


@dataclass(frozen=True)
class LabConfig:
    """Effective config used by every ``pypoe.lab.*`` module."""

    api_url: str = "http://localhost:8000"
    slack: SlackSection = field(default_factory=SlackSection)
    alerts: AlertsSection = field(default_factory=AlertsSection)
    mcp: McpSection = field(default_factory=McpSection)
    consult: ConsultSection = field(default_factory=ConsultSection)
    assistant: AssistantSection = field(default_factory=AssistantSection)
    dashboard: DashboardSection = field(default_factory=DashboardSection)

    # Where the YAML actually came from (None if no file was found).
    source_path: Optional[Path] = None


# ---------------------------------------------------------------------------


def _package_config_dir() -> Path:
    """Return the directory holding the lab-integration YAML configs.

    Lives next to ``pypoe.lab.config`` inside the installed package, at
    ``src/pypoe/config/`` in an editable install. Keeps user-edited
    config tucked away from the project root.
    """
    return Path(__file__).resolve().parent.parent / "config"


def _candidate_paths() -> list[Path]:
    """All YAML locations to try, in priority order.

    If ``PYPOE_LAB_CONFIG`` is set, it is **authoritative** — we do not
    fall back to the packaged location, so tests can disable YAML
    loading by pointing this env var at a non-existent path.
    """
    custom = os.environ.get("PYPOE_LAB_CONFIG")
    if custom:
        return [Path(custom).expanduser().resolve()]
    return [_package_config_dir() / DEFAULT_FILENAME]


def _load_yaml_file() -> tuple[dict, Optional[Path]]:
    """Load the first YAML file that exists. Returns ``({}, None)`` if none."""
    try:
        import yaml  # noqa: WPS433 — optional at lab-extra level
    except ImportError as exc:
        logger.debug("PyYAML not installed; skipping slack.yaml: %s", exc)
        return {}, None

    for path in _candidate_paths():
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception as exc:
            logger.warning("Failed to parse %s: %s — falling back to env/defaults", path, exc)
            return {}, None
        if not isinstance(data, dict):
            logger.warning("%s did not parse to a mapping (got %s)", path, type(data).__name__)
            return {}, None
        return data, path
    return {}, None


def _dig(d: dict, *keys: str, default: Any = None) -> Any:
    """Walk a nested dict; return ``default`` for missing branches."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _env_float(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r", name, raw)
        return None


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r", name, raw)
        return None


def _env_bool(name: str) -> Optional[bool]:
    """Parse a truthy/falsy env var. None when unset/empty."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> Optional[tuple[str, ...]]:
    """Comma-separated env var → tuple of stripped non-empty strings."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    items = tuple(s.strip() for s in raw.split(",") if s.strip())
    return items


def load_config() -> LabConfig:
    """Return the effective :class:`LabConfig` for this process.

    Reads ``slack.yaml`` (if present) and applies env-var overrides.
    Safe to call from any module — cached on first call.
    """
    if _LOADED["cfg"] is not None:
        return _LOADED["cfg"]

    raw, src = _load_yaml_file()

    lab_root = raw.get("lab") if isinstance(raw.get("lab"), dict) else raw

    api_url = (
        os.environ.get("LAB_API_URL")
        or _dig(lab_root, "api_url")
        or "http://localhost:8000"
    )

    slack = SlackSection(
        alert_channel=(
            os.environ.get("LAB_SLACK_CHANNEL")
            or _dig(lab_root, "slack", "alert_channel")
            or SlackSection.__dataclass_fields__["alert_channel"].default
        ),
        command_prefix=(
            os.environ.get("LAB_SLACK_COMMAND_PREFIX")
            or _dig(lab_root, "slack", "command_prefix")
            or SlackSection.__dataclass_fields__["command_prefix"].default
        ),
    )

    alerts = AlertsSection(
        max_concurrent_investigations=(
            _env_int("LAB_ALERT_MAX_CONCURRENT")
            or _dig(lab_root, "alerts", "max_concurrent_investigations")
            or AlertsSection.__dataclass_fields__["max_concurrent_investigations"].default
        ),
        investigation_model=(
            os.environ.get("LAB_INVESTIGATION_MODEL")
            or _dig(lab_root, "alerts", "investigation_model")
            or AlertsSection.__dataclass_fields__["investigation_model"].default
        ),
        investigation_reasoning_effort=(
            os.environ.get("LAB_INVESTIGATION_REASONING_EFFORT")
            or _dig(lab_root, "alerts", "investigation_reasoning_effort")
            or AlertsSection.__dataclass_fields__["investigation_reasoning_effort"].default
        ),
        investigation_timeout_s=(
            _env_float("LAB_INVESTIGATION_TIMEOUT_S")
            or _dig(lab_root, "alerts", "investigation_timeout_s")
            or AlertsSection.__dataclass_fields__["investigation_timeout_s"].default
        ),
    )

    mcp = McpSection(
        agent_source=(
            os.environ.get("LAB_MCP_AGENT_SOURCE")
            or _dig(lab_root, "mcp", "agent_source")
            or McpSection.__dataclass_fields__["agent_source"].default
        ),
        http_timeout_s=(
            _env_float("LAB_MCP_HTTP_TIMEOUT")
            or _dig(lab_root, "mcp", "http_timeout_s")
            or McpSection.__dataclass_fields__["http_timeout_s"].default
        ),
    )

    env_enabled = _env_bool("LAB_CONSULT_ENABLED")
    yaml_enabled = _dig(lab_root, "consult", "enabled")
    if env_enabled is not None:
        consult_enabled = env_enabled
    elif isinstance(yaml_enabled, bool):
        consult_enabled = yaml_enabled
    else:
        consult_enabled = ConsultSection.__dataclass_fields__["enabled"].default

    env_models = _env_list("LAB_CONSULT_MODELS")
    yaml_models = _dig(lab_root, "consult", "models")
    if env_models is not None:
        consult_models: tuple[str, ...] = env_models
    elif isinstance(yaml_models, list):
        consult_models = tuple(m for m in yaml_models if isinstance(m, str) and m.strip())
    else:
        consult_models = ConsultSection.__dataclass_fields__["models"].default

    consult = ConsultSection(enabled=consult_enabled, models=consult_models)

    env_monitor = _env_bool("LAB_ASSISTANT_MONITOR_ENABLED")
    yaml_monitor = _dig(lab_root, "assistant", "monitor_enabled")
    if env_monitor is not None:
        monitor_enabled = env_monitor
    elif isinstance(yaml_monitor, bool):
        monitor_enabled = yaml_monitor
    else:
        monitor_enabled = AssistantSection.__dataclass_fields__["monitor_enabled"].default

    assistant = AssistantSection(
        monitor_enabled=monitor_enabled,
        health_url=(
            os.environ.get("LAB_ASSISTANT_HEALTH_URL")
            or _dig(lab_root, "assistant", "health_url")
            or AssistantSection.__dataclass_fields__["health_url"].default
        ),
        probe_interval_s=(
            _env_float("LAB_ASSISTANT_PROBE_INTERVAL_S")
            or _dig(lab_root, "assistant", "probe_interval_s")
            or AssistantSection.__dataclass_fields__["probe_interval_s"].default
        ),
        service_name=(
            os.environ.get("LAB_ASSISTANT_SERVICE")
            or _dig(lab_root, "assistant", "service_name")
            or AssistantSection.__dataclass_fields__["service_name"].default
        ),
        env_root=(
            os.environ.get("LAB_ASSISTANT_ENV_ROOT")
            or _dig(lab_root, "assistant", "env_root")
            or AssistantSection.__dataclass_fields__["env_root"].default
        ),
        restart_wait_s=(
            _env_float("LAB_ASSISTANT_RESTART_WAIT_S")
            or _dig(lab_root, "assistant", "restart_wait_s")
            or AssistantSection.__dataclass_fields__["restart_wait_s"].default
        ),
        failures_to_alert=(
            _env_int("LAB_ASSISTANT_FAILURES_TO_ALERT")
            or _dig(lab_root, "assistant", "failures_to_alert")
            or AssistantSection.__dataclass_fields__["failures_to_alert"].default
        ),
        confirm_wait_s=(
            _env_float("LAB_ASSISTANT_CONFIRM_WAIT_S")
            or _dig(lab_root, "assistant", "confirm_wait_s")
            or AssistantSection.__dataclass_fields__["confirm_wait_s"].default
        ),
    )

    env_dash_monitor = _env_bool("LAB_DASHBOARD_MONITOR_ENABLED")
    yaml_dash_monitor = _dig(lab_root, "dashboard", "monitor_enabled")
    if env_dash_monitor is not None:
        dash_monitor_enabled = env_dash_monitor
    elif isinstance(yaml_dash_monitor, bool):
        dash_monitor_enabled = yaml_dash_monitor
    else:
        dash_monitor_enabled = DashboardSection.__dataclass_fields__["monitor_enabled"].default

    env_dash_paths = _env_list("LAB_DASHBOARD_PATHS")
    yaml_dash_paths = _dig(lab_root, "dashboard", "paths")
    if env_dash_paths is not None:
        dash_paths: tuple[str, ...] = env_dash_paths
    elif isinstance(yaml_dash_paths, list):
        dash_paths = tuple(p for p in yaml_dash_paths if isinstance(p, str) and p.strip())
    else:
        dash_paths = DashboardSection.__dataclass_fields__["paths"].default

    dashboard = DashboardSection(
        monitor_enabled=dash_monitor_enabled,
        base_url=(
            os.environ.get("LAB_DASHBOARD_BASE_URL")
            or _dig(lab_root, "dashboard", "base_url")
            or DashboardSection.__dataclass_fields__["base_url"].default
        ),
        paths=dash_paths,
        probe_interval_s=(
            _env_float("LAB_DASHBOARD_PROBE_INTERVAL_S")
            or _dig(lab_root, "dashboard", "probe_interval_s")
            or DashboardSection.__dataclass_fields__["probe_interval_s"].default
        ),
        service_name=(
            os.environ.get("LAB_DASHBOARD_SERVICE")
            or _dig(lab_root, "dashboard", "service_name")
            or DashboardSection.__dataclass_fields__["service_name"].default
        ),
        restart_wait_s=(
            _env_float("LAB_DASHBOARD_RESTART_WAIT_S")
            or _dig(lab_root, "dashboard", "restart_wait_s")
            or DashboardSection.__dataclass_fields__["restart_wait_s"].default
        ),
        expected_status=(
            _env_int("LAB_DASHBOARD_EXPECTED_STATUS")
            or _dig(lab_root, "dashboard", "expected_status")
            or DashboardSection.__dataclass_fields__["expected_status"].default
        ),
        failures_to_alert=(
            _env_int("LAB_DASHBOARD_FAILURES_TO_ALERT")
            or _dig(lab_root, "dashboard", "failures_to_alert")
            or DashboardSection.__dataclass_fields__["failures_to_alert"].default
        ),
        confirm_wait_s=(
            _env_float("LAB_DASHBOARD_CONFIRM_WAIT_S")
            or _dig(lab_root, "dashboard", "confirm_wait_s")
            or DashboardSection.__dataclass_fields__["confirm_wait_s"].default
        ),
    )

    cfg = LabConfig(
        api_url=api_url,
        slack=slack,
        alerts=alerts,
        mcp=mcp,
        consult=consult,
        assistant=assistant,
        dashboard=dashboard,
        source_path=src,
    )
    _LOADED["cfg"] = cfg
    return cfg


def reload_config() -> LabConfig:
    """Force re-read of the YAML + env. Useful in tests and in long-lived
    services that want to pick up a hot-edit without restarting.
    """
    _LOADED["cfg"] = None
    return load_config()


# Lazy singleton storage. A dict (not a module-level Optional[LabConfig])
# so tests can monkeypatch through `_LOADED["cfg"] = None`.
_LOADED: dict[str, Optional[LabConfig]] = {"cfg": None}
