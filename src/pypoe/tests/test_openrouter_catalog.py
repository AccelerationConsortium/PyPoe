"""Catalog discovery must preserve provider routing and survive outages."""
import asyncio
from types import SimpleNamespace

import httpx
import pytest

from pypoe.core import models


@pytest.fixture
def catalog(monkeypatch):
    monkeypatch.setattr(models, 'CHAT_MODELS', ['router/old', 'Poe-Bot'])
    monkeypatch.setattr(models, 'DEFAULT_CHAT_MODEL', 'router/old')
    monkeypatch.setattr(models, 'MODEL_PROVIDERS', {'router/old': 'openrouter'})
    monkeypatch.setattr(models, 'MODEL_PRICING_USD_PER_1M_TOKENS', {
        'router/old': {'prompt': 1, 'completion': 2},
        'Poe-Bot': {'prompt': 3, 'completion': 4},
    })
    monkeypatch.setattr(models, '_catalog_lock', asyncio.Lock())
    monkeypatch.setattr(models, '_catalog_key', None)
    monkeypatch.setattr(models, '_catalog_next_refresh', 0)
    return models


def row(model, output='text', prompt='0.000002'):
    return {'id': model, 'architecture': {'output_modalities': [output]},
            'pricing': {'prompt': prompt, 'completion': '0.00001'}}


def test_replaces_choices_but_preserves_historical_routing(catalog):
    shared = catalog.CHAT_MODELS
    catalog._apply_openrouter_catalog({'data': [row('router/new'), row('image', 'image')]})
    assert shared == ['router/new', 'Poe-Bot']
    assert catalog.provider_for('router/new') == 'openrouter'
    assert catalog.provider_for('router/old') == 'openrouter'
    assert catalog.provider_for('Poe-Bot') == 'poe'
    assert catalog.MODEL_PRICING_USD_PER_1M_TOKENS['router/new'] == {'prompt': 2, 'completion': 10}
    assert catalog.MODEL_PRICING_USD_PER_1M_TOKENS['Poe-Bot']['prompt'] == 3


def test_preserves_default_deduplicates_and_rejects_price_nan(catalog):
    catalog._apply_openrouter_catalog({'data': [row('router/new'), row('router/old', prompt='nan'),
                                                row('router/new'), row('Poe-Bot')]})
    assert catalog.CHAT_MODELS == ['router/old', 'router/new', 'Poe-Bot']
    assert 'router/old' not in catalog.MODEL_PRICING_USD_PER_1M_TOKENS
    assert catalog.provider_for('Poe-Bot') == 'poe'


@pytest.mark.parametrize('payload', [{}, {'data': None}, {'data': [row('new'), {}]}])
def test_invalid_payload_does_not_partially_update(catalog, payload):
    with pytest.raises(ValueError):
        catalog._apply_openrouter_catalog(payload)
    assert catalog.CHAT_MODELS == ['router/old', 'Poe-Bot']


def test_empty_allowed_catalog_removes_openrouter_choices(catalog):
    catalog._apply_openrouter_catalog({'data': []})
    assert catalog.CHAT_MODELS == ['Poe-Bot']
    assert catalog.provider_for('router/old') == 'openrouter'


@pytest.mark.asyncio
async def test_refresh_caches_concurrent_requests_and_retries_failure(catalog, monkeypatch):
    calls = []
    now = [100]
    monkeypatch.setattr(catalog.time, 'monotonic', lambda: now[0])
    def handler(request):
        calls.append(request)
        assert str(request.url) == 'https://openrouter.ai/api/v1/models/user'
        assert request.headers['Authorization'] == 'Bearer test-key'
        if len(calls) == 1:
            return httpx.Response(200, json={'data': [row('router/new')]})
        return httpx.Response(503)
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: original(
        transport=httpx.MockTransport(handler), **kw))
    await asyncio.gather(*(catalog.refresh_openrouter_catalog('test-key') for _ in range(3)))
    assert len(calls) == 1
    now[0] += 3601
    assert not await catalog.refresh_openrouter_catalog('test-key')
    assert catalog.CHAT_MODELS == ['router/new', 'Poe-Bot']
    assert not await catalog.refresh_openrouter_catalog('test-key')
    assert len(calls) == 2
    now[0] += 61
    await catalog.refresh_openrouter_catalog('test-key')
    assert len(calls) == 3
    assert not await catalog.refresh_openrouter_catalog('')


@pytest.mark.asyncio
async def test_client_refreshes_before_returning_choices(monkeypatch):
    from pypoe.core import client
    calls = []
    async def refresh(key):
        calls.append(key)
    monkeypatch.setattr(client, 'refresh_openrouter_catalog', refresh)
    obj = object.__new__(client.PoeChatClient)
    obj.config = SimpleNamespace(openrouter_api_key='test-key')
    assert await obj.get_available_bots() == list(client.CHAT_MODELS)
    assert calls == ['test-key']
    obj.config.openrouter_auto_refresh_models = False
    await obj.get_available_bots()
    assert calls == ['test-key']
