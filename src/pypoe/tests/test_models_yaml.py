"""Tests for ``pypoe.core.models``'s YAML loader.

Validates the YAML→fallback precedence by pointing
``PYPOE_MODELS_CONFIG`` at a tmp file and re-importing the module.
"""

from __future__ import annotations

import importlib
import sys

import pytest

yaml = pytest.importorskip("yaml")


def _reimport(monkeypatch, models_path):
    """Re-import pypoe.core.models with PYPOE_MODELS_CONFIG pointed at path."""
    monkeypatch.setenv("PYPOE_MODELS_CONFIG", str(models_path))
    # Drop the cached module so the loader runs again with the new env.
    sys.modules.pop("pypoe.core.models", None)
    return importlib.import_module("pypoe.core.models")


def test_yaml_overrides_hardcoded(tmp_path, monkeypatch):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(yaml.safe_dump({
        "default": "Test-Bot-1",
        "chat_models": ["Test-Bot-1", "Test-Bot-2"],
        "pricing_usd_per_1m_tokens": {
            "Test-Bot-1": {"prompt": 1.0, "completion": 2.0},
        },
    }))
    mod = _reimport(monkeypatch, cfg)
    assert mod.CHAT_MODELS == ["Test-Bot-1", "Test-Bot-2"]
    assert mod.DEFAULT_CHAT_MODEL == "Test-Bot-1"
    assert mod.MODEL_PRICING_USD_PER_1M_TOKENS["Test-Bot-1"]["prompt"] == 1.0


def test_missing_file_uses_fallbacks(tmp_path, monkeypatch):
    """If models.yaml doesn't exist, the hardcoded fallbacks ship."""
    mod = _reimport(monkeypatch, tmp_path / "does_not_exist.yaml")
    # Hardcoded snapshot includes Claude-Opus-4.8 as default.
    assert mod.DEFAULT_CHAT_MODEL == "Claude-Opus-4.8"
    assert "Claude-Opus-4.7" in mod.CHAT_MODELS


def test_malformed_yaml_uses_fallbacks(tmp_path, monkeypatch):
    """Parse errors fall back rather than crashing import."""
    cfg = tmp_path / "models.yaml"
    cfg.write_text("not: valid: yaml::: at all")
    mod = _reimport(monkeypatch, cfg)
    assert mod.DEFAULT_CHAT_MODEL == "Claude-Opus-4.8"


def test_partial_yaml_fills_in_missing_keys(tmp_path, monkeypatch):
    """A YAML with only `chat_models` keeps fallback default + pricing."""
    cfg = tmp_path / "models.yaml"
    cfg.write_text(yaml.safe_dump({"chat_models": ["Only-Bot"]}))
    mod = _reimport(monkeypatch, cfg)
    assert mod.CHAT_MODELS == ["Only-Bot"]
    assert mod.DEFAULT_CHAT_MODEL == "Claude-Opus-4.8"  # fallback
    assert "Claude-Opus-4.7" in mod.MODEL_PRICING_USD_PER_1M_TOKENS  # fallback


@pytest.fixture(autouse=True)
def _reset_models_module_after(monkeypatch):
    """Restore the production-config-loaded module after each test."""
    yield
    monkeypatch.delenv("PYPOE_MODELS_CONFIG", raising=False)
    sys.modules.pop("pypoe.core.models", None)
    importlib.import_module("pypoe.core.models")


# ---------------------------------------------------------------------------
# The shipped roster (config/models.yaml + models.example.yaml)
# ---------------------------------------------------------------------------

def _shipped(name):
    """Load a roster file from src/pypoe/config/.

    ``models.yaml`` is gitignored local config, so it is absent on a fresh
    clone and in CI — skip rather than fail there. ``models.example.yaml`` is
    committed and must always be present, so a missing one is a real error.
    """
    from pathlib import Path
    import pypoe.core.models as m

    path = Path(m.__file__).resolve().parent.parent / "config" / name
    if not path.is_file():
        if name == "models.yaml":
            pytest.skip("models.yaml is local config; absent on a fresh clone")
        raise AssertionError(f"committed roster {name} is missing")
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("filename", ["models.yaml", "models.example.yaml"])
def test_shipped_roster_has_routed_and_priced_default(filename):
    """Local catalogs can refresh independently of the example snapshot."""
    from pypoe.core.models import _parse_chat_models

    data = _shipped(filename)
    ids, providers = _parse_chat_models(data["chat_models"])
    default = data["default"]
    assert default in ids
    assert providers[default] == "openrouter"
    pricing = data["pricing_usd_per_1m_tokens"]
    for model in ids:
        if providers.get(model) == "openrouter":
            assert pricing[model]["prompt"] >= 0
            assert pricing[model]["completion"] >= 0


@pytest.mark.parametrize("filename", ["models.yaml", "models.example.yaml"])
def test_shipped_roster_has_no_duplicate_ids(filename):
    from pypoe.core.models import _parse_chat_models

    ids, _ = _parse_chat_models(_shipped(filename)["chat_models"])
    assert len(ids) == len(set(ids)), f"duplicate model ids in {filename}: {ids}"
