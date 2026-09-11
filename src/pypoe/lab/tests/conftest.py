"""Shared pytest fixtures for ``pypoe.lab`` tests.

Every test runs against a *fresh, empty* lab config so that the host's
real ``slack.yaml`` and ``.env`` don't leak in. Tests that want
non-default values opt in via ``monkeypatch.setenv`` or by writing
their own YAML and pointing ``PYPOE_LAB_CONFIG`` at it.
"""

from __future__ import annotations

import pytest

from pypoe.lab import config as lab_config


_LAB_ENV_VARS = (
    "LAB_API_URL",
    "LAB_SLACK_CHANNEL",
    "LAB_SLACK_COMMAND_PREFIX",
    "LAB_ALERT_MAX_CONCURRENT",
    "LAB_INVESTIGATION_MODEL",
    "LAB_INVESTIGATION_REASONING_EFFORT",
    "LAB_INVESTIGATION_TIMEOUT_S",
    "LAB_MCP_AGENT_SOURCE",
    "LAB_MCP_HTTP_TIMEOUT",
    "PYPOE_LAB_CONFIG",
)


@pytest.fixture(autouse=True)
def _isolate_lab_config(monkeypatch):
    for var in _LAB_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # Point the loader at a path that doesn't exist so it can't find
    # the repo's real slack.yaml.
    monkeypatch.setenv("PYPOE_LAB_CONFIG", "/nonexistent/slack.yaml")
    lab_config.reload_config()
    yield
    lab_config.reload_config()
