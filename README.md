# PyPoe

A Python chat client that exposes models from **Poe** and **OpenRouter**
through three interfaces — a CLI, a local web UI, and a Slack bot — all sharing
a single SQLite conversation history at `~/.pypoe/single_webchat_history.db`.

Routing is **per model**: each entry in the model catalog names its provider, so
a Poe model and an OpenRouter model can be used side by side, including as the
two sides of a debate. At least one provider key is required; it need not be
Poe. See [Model providers](#model-providers).

## Three interfaces

- **CLI.** Interactive chat from the terminal.

  ```bash
  pypoe cli select          # pick or start a conversation
  pypoe cli chat            # start/continue a chat
  pypoe cli list            # list conversations
  ```

- **Web UI.** Browser chat with a sidebar of past conversations.

  ```bash
  pypoe web                 # http://127.0.0.1:8000
  ```

- **Slack bot.** `/poe` slash command, `@PyPoe` mentions, and DMs, with
  per-thread context.

  ```bash
  pypoe slack
  ```

All three write to and read from the same database, so conversations
started in one interface continue in the others.

### Optional: lab integration

PyPoe also ships a read-only **lab interface layer** for the AC Organic
Self-driving Lab. With `pip install -e ".[lab]"` you additionally get:

- `pypoe lab-mcp` — a read-only MCP server (talk to the lab from
  Claude Desktop / Code).
- `pypoe lab-status` — one-shot aggregator health summary.
- `/lab-*` Slack slash commands (auto-registered on `pypoe slack` when
  `LAB_API_URL` is set).
- `POST /alerts/kuma` + `POST /alerts/device` webhooks (auto-mounted on
  `pypoe web` under the same condition): Uptime Kuma service alerts and
  aggregator-pushed device alerts, both posted to Slack with a threaded
  `claude -p` investigation.
- `GET /kuma/status` — a STATUS_SPEC envelope gateway-fronting Uptime
  Kuma so the lab dashboard can show an alerting-watchdog tile.

The integration is **read-only at the device level**: there is no
`control_action` tool and no `/control/*` calls. Control flows through
the `lab-skills` SDK in `ac-organic-lab/`. See
[docs/LAB_INTEGRATION.md](docs/LAB_INTEGRATION.md) for setup.

## Install

PyPoe is editable-installed from a clone:

```bash
git clone https://github.com/your-username/pypoe.git
cd pypoe
pip install -e .                  # CLI only
pip install -e ".[web-ui]"        # CLI + web + Slack
pip install -e ".[dev]"           # everything + test/lint tools
```

The web UI and Slack bot share the same `web-ui` extra (both depend on
`fastapi`, `slack-bolt`, etc.). Image/video auto-download is a separate
`media` extra and requires `PYPOE_ENABLE_MEDIA=true` at runtime.

## Quickstart

1. Get a key from at least one provider — [OpenRouter](https://openrouter.ai/keys)
   or [Poe](https://poe.com/api_key).
2. Create a `.env` in the repo root:

   ```env
   OPENROUTER_API_KEY=sk-or-...
   POE_API_KEY=your-poe-api-key      # optional; only for Poe-routed models
   ```

3. Pick an interface:

   ```bash
   pypoe cli select   # terminal
   pypoe web          # browser at http://127.0.0.1:8000
   pypoe slack        # requires Slack app + SLACK_* env vars
   ```

For network access, authentication, Slack app setup, or running as a
service, see the topic-specific docs below.

## Model providers

The fallback catalog and default live in `src/pypoe/config/models.yaml` (gitignored; copy from
`models.example.yaml`). Each entry names its provider — a bare string is a Poe
model, a mapping is anything else:

```yaml
default: z-ai/glm-5.2

chat_models:
  - {id: z-ai/glm-5.2, provider: openrouter}
  - {id: deepseek/deepseek-v4-flash-0731, provider: openrouter}
  - Claude-Opus-4.8            # bare string => Poe
```

With `OPENROUTER_API_KEY` set, opening the model picker refreshes OpenRouter
choices from `/api/v1/models/user`, filtered by that key's provider preferences
and guardrails. Results are cached in memory for one hour; failed refreshes keep
the previous catalog and retry after one minute. Poe entries stay configured in
YAML. The configured default stays first when available, and existing chats keep
their selected model. Base token prices refresh too; tiered pricing and extra
charges are not included in these estimates.

Set `PYPOE_OPENROUTER_AUTO_REFRESH_MODELS=false` to use only the YAML catalog.
YAML/default changes require a service restart; automatic discovery does not
rewrite the file. `scripts/utils/update_models.py` is a legacy availability
tester, not a catalog discovery tool.

| Provider | Key | Billing | Notes |
|---|---|---|---|
| `openrouter` | `OPENROUTER_API_KEY` | **per token** | ~400 models; ids must match [OpenRouter's slugs](https://openrouter.ai/models) exactly |
| `poe` | `POE_API_KEY` | flat subscription | the only provider with image/video generation bots |

A provider with no key configured is skipped: its models report a clear
"not configured" error if selected, and `/status` omits it rather than calling
it unhealthy. The service is healthy as long as **one** provider can answer, so
a lapsed Poe subscription alongside a working OpenRouter key is not an outage.

**Spend guards** (OpenRouter only, since Poe is a flat subscription):

```env
PYPOE_OPENROUTER_MAX_TOKENS=4096   # ceiling on every completion; 0 disables
PYPOE_OPENROUTER_MIN_CREDITS=1.0   # USD below which /status reports degraded
```

`GET /status` carries one component per configured provider (`poe_api`,
`openrouter_api`) with a provider-qualified error code — `poe_subscription_required`,
`openrouter_auth_failed` — plus the remaining OpenRouter balance. Health is
observed from real traffic, with a `max_tokens=1` probe only when that evidence
goes stale.

## Documentation

- [docs/README_SETUP.md](docs/README_SETUP.md) — install options, `.env`
  layout, environment variables, basic troubleshooting.
- [docs/README_WEB.md](docs/README_WEB.md) — running the web UI:
  network access, authentication, bot locking.
- [docs/README_SLACK.md](docs/README_SLACK.md) — Slack app creation,
  scopes, slash command and events, per-thread conversation scoping.
- [docs/README_SYSTEMD.md](docs/README_SYSTEMD.md) — running `pypoe-web`
  and `pypoe-slack` as user-level systemd services.
- [docs/README_HISTORY.md](docs/README_HISTORY.md) — history database:
  where it lives, schema, Slack id scheme, sqlite cookbook, cleanup
  commands.
- [docs/LAB_INTEGRATION.md](docs/LAB_INTEGRATION.md) — AC Organic
  Self-driving Lab integration: MCP server, `/lab-*` slash commands,
  Kuma alert webhook, environment variables, and the read-only
  guard-rails (no `/control/*`).

## Project layout

```
pypoe/
  src/pypoe/
    core/          # client, history, config, cli entry point
    interfaces/    # web/, slack/, cli/
    lab/           # optional ac-organic-lab integration (read-only MCP,
                   #   /lab-* Slack commands, /alerts/kuma webhook)
    config/        # slack.yaml + models.yaml (lab + model catalog)
    tests/
  scripts/         # ops + setup helpers
  docs/            # topic-specific documentation
  examples/        # SDK usage samples
```

## License

MIT — see [LICENSE](LICENSE).

## Support

- Issues: <https://github.com/AccelerationConsortium/pypoe/issues>
- Poe API reference: <https://creator.poe.com/docs/quick-start>
