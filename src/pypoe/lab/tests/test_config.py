"""Unit tests for ``pypoe.lab.config``.

Covers the YAML+env precedence rules and the "no YAML, no env vars"
fallback path. Uses ``PYPOE_LAB_CONFIG`` to point the loader at a
tmp_path file so tests are hermetic.
"""

from __future__ import annotations

import pytest

yaml = pytest.importorskip("yaml")

from pypoe.lab import config as lab_config

# The shared autouse fixture in conftest.py already resets env vars and
# points the loader at a non-existent path.


def _write_yaml(tmp_path, data: dict):
    p = tmp_path / "slack.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_defaults_when_no_yaml_no_env(monkeypatch):
    cfg = lab_config.load_config()
    assert cfg.api_url == "http://localhost:8000"
    assert cfg.slack.alert_channel == "#lab-alerts"
    assert cfg.slack.command_prefix == "/lab-"
    assert cfg.alerts.max_concurrent_investigations == 2
    assert cfg.mcp.agent_source == "claude-agent"
    assert cfg.mcp.http_timeout_s == 10.0
    assert cfg.source_path is None


def test_yaml_values_override_defaults(monkeypatch, tmp_path):
    path = _write_yaml(tmp_path, {
        "lab": {
            "api_url": "http://lab.example:8001",
            "slack": {"alert_channel": "#lab-x", "command_prefix": "/xyz-lab-"},
            "alerts": {"max_concurrent_investigations": 4},
            "mcp": {"agent_source": "agent-x", "http_timeout_s": 30},
        }
    })
    monkeypatch.setenv("PYPOE_LAB_CONFIG", str(path))
    cfg = lab_config.reload_config()
    assert cfg.api_url == "http://lab.example:8001"
    assert cfg.slack.alert_channel == "#lab-x"
    assert cfg.slack.command_prefix == "/xyz-lab-"
    assert cfg.alerts.max_concurrent_investigations == 4
    assert cfg.mcp.agent_source == "agent-x"
    assert cfg.mcp.http_timeout_s == 30.0
    assert cfg.source_path == path


def test_env_var_beats_yaml(monkeypatch, tmp_path):
    path = _write_yaml(tmp_path, {
        "lab": {"slack": {"command_prefix": "/yaml-lab-"}}
    })
    monkeypatch.setenv("PYPOE_LAB_CONFIG", str(path))
    monkeypatch.setenv("LAB_SLACK_COMMAND_PREFIX", "/env-lab-")
    cfg = lab_config.reload_config()
    assert cfg.slack.command_prefix == "/env-lab-"


def test_partial_yaml_uses_defaults_for_missing_keys(monkeypatch, tmp_path):
    path = _write_yaml(tmp_path, {
        "lab": {"slack": {"alert_channel": "#only-this"}}
    })
    monkeypatch.setenv("PYPOE_LAB_CONFIG", str(path))
    cfg = lab_config.reload_config()
    assert cfg.slack.alert_channel == "#only-this"
    # Untouched keys → defaults.
    assert cfg.slack.command_prefix == "/lab-"
    assert cfg.alerts.max_concurrent_investigations == 2
    assert cfg.api_url == "http://localhost:8000"


def test_malformed_yaml_falls_back_to_defaults(monkeypatch, tmp_path):
    path = tmp_path / "slack.yaml"
    path.write_text("not: valid: yaml: at all:::")
    monkeypatch.setenv("PYPOE_LAB_CONFIG", str(path))
    cfg = lab_config.reload_config()
    assert cfg.api_url == "http://localhost:8000"
    assert cfg.source_path is None  # marked as "didn't load"


def test_non_mapping_yaml_falls_back(monkeypatch, tmp_path):
    path = tmp_path / "slack.yaml"
    path.write_text("- just\n- a\n- list\n")
    monkeypatch.setenv("PYPOE_LAB_CONFIG", str(path))
    cfg = lab_config.reload_config()
    assert cfg.api_url == "http://localhost:8000"
    assert cfg.source_path is None


def test_yaml_without_lab_key_treated_as_flat(monkeypatch, tmp_path):
    """`slack.yaml` may use either {lab: {...}} or a flat top-level shape."""
    path = _write_yaml(tmp_path, {
        "slack": {"command_prefix": "/flat-"}
    })
    monkeypatch.setenv("PYPOE_LAB_CONFIG", str(path))
    cfg = lab_config.reload_config()
    assert cfg.slack.command_prefix == "/flat-"


def test_non_numeric_env_int_is_ignored(monkeypatch):
    monkeypatch.setenv("LAB_ALERT_MAX_CONCURRENT", "not-a-number")
    cfg = lab_config.reload_config()
    assert cfg.alerts.max_concurrent_investigations == 2  # default


def test_consult_defaults(monkeypatch):
    cfg = lab_config.load_config()
    assert cfg.consult.enabled is True
    assert cfg.consult.models == lab_config.ConsultSection().models


def test_consult_defaults_are_reachable_models():
    """Every default consult model must be in the catalog.

    A name absent from ``chat_models`` still resolves — to the default
    provider — and only fails at call time, inside an alert investigation
    where nobody is watching. The previous defaults (``GPT-5.4``, ``GLM-5.2``)
    were Poe-routed and silently became unreachable when that subscription
    lapsed.
    """
    from pypoe.core.models import CHAT_MODELS

    for model in lab_config.ConsultSection().models:
        assert model in CHAT_MODELS, f"consult model {model!r} is not in chat_models"


def test_investigation_model_default_is_luna_max(monkeypatch):
    cfg = lab_config.load_config()
    assert cfg.alerts.investigation_model == "gpt-5.6-luna"
    assert cfg.alerts.investigation_reasoning_effort == "max"


def test_consult_yaml(monkeypatch, tmp_path):
    path = _write_yaml(tmp_path, {
        "lab": {
            "consult": {
                "enabled": False,
                "models": ["Claude-Sonnet-4.6", "Gemini-3.1-Pro"],
            }
        }
    })
    monkeypatch.setenv("PYPOE_LAB_CONFIG", str(path))
    cfg = lab_config.reload_config()
    assert cfg.consult.enabled is False
    assert cfg.consult.models == ("Claude-Sonnet-4.6", "Gemini-3.1-Pro")


def test_consult_env_overrides_yaml(monkeypatch, tmp_path):
    path = _write_yaml(tmp_path, {
        "lab": {"consult": {"enabled": True, "models": ["A", "B"]}}
    })
    monkeypatch.setenv("PYPOE_LAB_CONFIG", str(path))
    monkeypatch.setenv("LAB_CONSULT_ENABLED", "false")
    monkeypatch.setenv("LAB_CONSULT_MODELS", "X, Y , Z ,")
    cfg = lab_config.reload_config()
    assert cfg.consult.enabled is False
    assert cfg.consult.models == ("X", "Y", "Z")   # empty entry dropped


def test_consult_empty_yaml_models_yields_empty(monkeypatch, tmp_path):
    """An explicit empty list should be honoured, not silently re-defaulted."""
    path = _write_yaml(tmp_path, {"lab": {"consult": {"models": []}}})
    monkeypatch.setenv("PYPOE_LAB_CONFIG", str(path))
    cfg = lab_config.reload_config()
    assert cfg.consult.models == ()


def test_load_is_cached_until_reload(monkeypatch):
    cfg1 = lab_config.load_config()
    cfg2 = lab_config.load_config()
    assert cfg1 is cfg2
    monkeypatch.setenv("LAB_SLACK_CHANNEL", "#changed")
    # load_config() returns cached, NOT the new env value.
    cfg3 = lab_config.load_config()
    assert cfg3 is cfg1
    # reload_config() re-reads.
    cfg4 = lab_config.reload_config()
    assert cfg4.slack.alert_channel == "#changed"


def test_dashboard_defaults(monkeypatch):
    cfg = lab_config.load_config()
    assert cfg.dashboard.monitor_enabled is False
    assert "/api/openapi.json" in cfg.dashboard.paths
    assert cfg.dashboard.expected_status == 200


def test_dashboard_yaml_and_env(monkeypatch, tmp_path):
    path = tmp_path / "slack.yaml"
    path.write_text(
        "lab:\n"
        "  dashboard:\n"
        "    monitor_enabled: true\n"
        "    base_url: http://127.0.0.1:9001\n"
        "    paths:\n"
        "      - /api/openapi.json\n"
        "      - /api/catalog\n"
    )
    monkeypatch.setenv("PYPOE_LAB_CONFIG", str(path))
    cfg = lab_config.reload_config()
    assert cfg.dashboard.monitor_enabled is True
    assert cfg.dashboard.base_url == "http://127.0.0.1:9001"
    assert cfg.dashboard.paths == ("/api/openapi.json", "/api/catalog")
    # env beats yaml
    monkeypatch.setenv("LAB_DASHBOARD_BASE_URL", "http://127.0.0.1:9101")
    cfg2 = lab_config.reload_config()
    assert cfg2.dashboard.base_url == "http://127.0.0.1:9101"


def test_investigation_reasoning_yaml_and_env(monkeypatch, tmp_path):
    path = _write_yaml(tmp_path, {"lab": {"alerts": {"investigation_reasoning_effort": "high"}}})
    monkeypatch.setenv("PYPOE_LAB_CONFIG", str(path))
    assert lab_config.reload_config().alerts.investigation_reasoning_effort == "high"
    monkeypatch.setenv("LAB_INVESTIGATION_REASONING_EFFORT", "max")
    assert lab_config.reload_config().alerts.investigation_reasoning_effort == "max"
