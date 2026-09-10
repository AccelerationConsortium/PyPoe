"""Chat-only Poe model catalog used by PyPoe interfaces.

The list lives in ``src/pypoe/config/models.yaml`` (gitignored; copy
from ``models.example.yaml`` next to it). This module loads it at
import time and exposes the same three constants the rest of the
codebase has always imported:

  * ``CHAT_MODELS``
  * ``DEFAULT_CHAT_MODEL``
  * ``MODEL_PRICING_USD_PER_1M_TOKENS``

If the YAML is absent / malformed, the hardcoded fallback below is
used so existing deployments keep working without action.

Override the file location with ``PYPOE_MODELS_CONFIG=<path>``.
"""

import asyncio
import hashlib
import logging
import math
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


_FALLBACK_DEFAULT = "Claude-Opus-4.8"

_FALLBACK_CHAT_MODELS = [
    "Claude-Opus-4.8",
    "Claude-Sonnet-4.6",
    "Claude-Opus-4.7",
    "GPT-5.4",
    "GPT-4-Turbo",
    "Grok-4",
    "Gemini-3.1-Pro",
    "Gemini-3-Flash",
    "GLM-5.2",
    "Kimi-K3",
]

# Snapshot from https://models.poecdn.net/models.json on 2026-07-19;
# every priced entry is live on Poe. Keep in step with
# config/models.example.yaml. Kimi-K3 is a live Poe bot not yet in the
# pricing feed, so it ships in the roster but carries no price entry.
_FALLBACK_PRICING = {
    "Claude-Opus-4.8":    {"prompt": 4.2929,  "completion": 21.4646},
    "Claude-Sonnet-4.6":  {"prompt": 2.5758,  "completion": 12.8788},
    "Claude-Opus-4.7":    {"prompt": 4.2929,  "completion": 21.4646},
    "GPT-5.4":            {"prompt": 2.2727,  "completion": 13.6364},
    "GPT-4-Turbo":        {"prompt": 9.0909,  "completion": 27.2727},
    "Grok-4":             {"prompt": 3.0303,  "completion": 15.1515},
    "Gemini-3.1-Pro":     {"prompt": 2.0202,  "completion": 12.1212},
    "Gemini-3-Flash":     {"prompt": 0.4040,  "completion": 2.4242},
    "GLM-5.2":            {"prompt": 1.4141,  "completion": 4.4444},
}


def _config_path() -> Path:
    """``src/pypoe/config/models.yaml`` (sibling of ``core/``)."""
    custom = os.environ.get("PYPOE_MODELS_CONFIG")
    if custom:
        return Path(custom).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / "config" / "models.yaml"


def _load() -> tuple[list[str], str, dict, dict]:
    """Read models.yaml; on any failure fall back to the hardcoded values."""
    _fallback = (_FALLBACK_CHAT_MODELS, _FALLBACK_DEFAULT, _FALLBACK_PRICING, {})

    path = _config_path()
    if not path.is_file():
        return _fallback

    try:
        import yaml  # PyYAML is a transitive dep of pypoe[web-ui]/[lab]
    except ImportError:
        logger.debug("PyYAML not available; using hardcoded model list")
        return _fallback

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.warning("Could not parse %s (%s) — using hardcoded model list", path, exc)
        return _fallback

    chat_models, providers = _parse_chat_models(data.get("chat_models"))

    default = data.get("default") if isinstance(data.get("default"), str) else _FALLBACK_DEFAULT
    pricing = data.get("pricing_usd_per_1m_tokens")
    if not isinstance(pricing, dict):
        pricing = _FALLBACK_PRICING

    return chat_models, default, pricing, providers


def _parse_chat_models(raw) -> tuple[list[str], dict[str, str]]:
    """Parse ``chat_models`` into ``(ids, {id: provider})``.

    Two entry forms are accepted, so existing rosters keep working untouched:

    * ``- Claude-Opus-4.8`` — a bare string, implicitly Poe.
    * ``- {id: anthropic/claude-opus-4.8, provider: openrouter}`` — mapping form.

    The model **id stays the single identity** everywhere else (UI dropdowns,
    the ``bot_name`` / ``model_name`` history columns, group/debate validation),
    so tagging a provider here adds a routing dimension without forcing a
    schema or API-shape change downstream.
    """
    if not isinstance(raw, list):
        return list(_FALLBACK_CHAT_MODELS), {}

    ids: list[str] = []
    providers: dict[str, str] = {}
    for entry in raw:
        if isinstance(entry, str):
            ids.append(entry)
        elif isinstance(entry, dict):
            model_id = entry.get("id") or entry.get("name")
            if not isinstance(model_id, str) or not model_id:
                logger.warning("Skipping chat_models entry without an id: %r", entry)
                continue
            ids.append(model_id)
            provider = entry.get("provider")
            if isinstance(provider, str) and provider.strip():
                providers[model_id] = provider.strip().lower()
        else:
            logger.warning("Skipping unrecognised chat_models entry: %r", entry)

    if not ids:
        return list(_FALLBACK_CHAT_MODELS), {}
    return ids, providers


(
    CHAT_MODELS,
    DEFAULT_CHAT_MODEL,
    MODEL_PRICING_USD_PER_1M_TOKENS,
    MODEL_PROVIDERS,
) = _load()


_catalog_lock = asyncio.Lock()
_catalog_key = None
_catalog_next_refresh = 0.0


def _apply_openrouter_catalog(payload: dict) -> None:
    """Replace OpenRouter choices, retaining routing for existing conversations."""
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ValueError("OpenRouter catalog must contain a data list")
    discovered = {}
    prices = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("OpenRouter catalog contains an invalid model")
        model = row["id"]
        architecture = row.get("architecture") or {}
        if "text" not in architecture.get("output_modalities", ["text"]):
            continue
        # An explicitly configured model belonging to another provider wins.
        if model in CHAT_MODELS and provider_for(model) != "openrouter":
            continue
        discovered[model] = None
        try:
            price = {kind: float(row["pricing"][kind]) * 1_000_000
                     for kind in ("prompt", "completion")}
            if all(math.isfinite(value) and value >= 0 for value in price.values()):
                prices[model] = price
        except (KeyError, TypeError, ValueError, OverflowError):
            pass  # Missing/variable prices display as unknown.

    other_models = [m for m in CHAT_MODELS if provider_for(m) != "openrouter"]
    choices = list(discovered) + other_models
    if DEFAULT_CHAT_MODEL in choices:
        choices.remove(DEFAULT_CHAT_MODEL)
        choices.insert(0, DEFAULT_CHAT_MODEL)
    # Mutate in place: interfaces import these shared objects directly.
    CHAT_MODELS[:] = choices
    MODEL_PROVIDERS.update({m: "openrouter" for m in discovered})
    for model in discovered:
        MODEL_PRICING_USD_PER_1M_TOKENS.pop(model, None)
    MODEL_PRICING_USD_PER_1M_TOKENS.update(prices)


async def refresh_openrouter_catalog(api_key: str, *, force: bool = False) -> bool:
    """Refresh on demand, at most hourly; failures keep the last usable catalog.

    Uses the key-filtered endpoint so workspace restrictions are respected.
    A failed refresh is retried after one minute, not on every picker request.
    The application uses one deployment-wide provider key, as with routing.
    """
    global _catalog_key, _catalog_next_refresh
    if not api_key:
        return False
    fingerprint = hashlib.sha256(api_key.encode()).digest()
    async with _catalog_lock:
        if not force and fingerprint == _catalog_key and time.monotonic() < _catalog_next_refresh:
            return False
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://openrouter.ai/api/v1/models/user",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                _apply_openrouter_catalog(response.json())
        except Exception as exc:
            # Do not log response bodies or credentials.
            logger.warning("OpenRouter catalog refresh failed (%s); retaining previous catalog",
                           type(exc).__name__)
            _catalog_key = fingerprint
            _catalog_next_refresh = time.monotonic() + 60
            return False
        _catalog_key = fingerprint
        _catalog_next_refresh = time.monotonic() + 3600
        return True


def provider_for(model: str) -> str:
    """Which provider serves ``model``.

    Falls back to the default provider (Poe) for anything unlisted, which keeps
    free-text callers working — the CLI's ``--bot`` and the lab MCP's
    ``consult_poe`` both accept models that are not in the roster.
    """
    from .providers import DEFAULT_PROVIDER

    return MODEL_PROVIDERS.get(model, DEFAULT_PROVIDER)


def models_by_provider() -> dict[str, list[str]]:
    """The roster grouped by provider, in roster order."""
    grouped: dict[str, list[str]] = {}
    for model in CHAT_MODELS:
        grouped.setdefault(provider_for(model), []).append(model)
    return grouped


def dollar_meter(rate_per_1m_tokens: float) -> str:
    """Return one '$' per $1.00 per 1M tokens."""
    if rate_per_1m_tokens <= 0:
        return "-"
    return "$" * max(1, math.ceil(rate_per_1m_tokens))


def get_model_price_markers(model: str) -> tuple[str, str]:
    """Return prompt/completion dollar-sign markers for a model."""
    pricing = MODEL_PRICING_USD_PER_1M_TOKENS.get(model)
    if not pricing:
        return ("?", "?")

    return (
        dollar_meter(pricing["prompt"]),
        dollar_meter(pricing["completion"]),
    )


def format_model_price_marker(model: str) -> str:
    """Format Slack-friendly prompt/completion price markers."""
    prompt, completion = get_model_price_markers(model)
    if prompt == "?" or completion == "?":
        return "price unknown"
    return f"in {prompt} / out {completion}"
