# PyPoe Background Services

PyPoe is run in production as two user-level systemd services:

- `pypoe-web.service` — the FastAPI web UI ([README_WEB.md](README_WEB.md)).
- `pypoe-slack.service` — the Slack Socket Mode bot ([README_SLACK.md](README_SLACK.md)).

Both load secrets from the repo-root `.env`, restart on failure, and
log to the user journal.

## Unit files

Save as `~/.config/systemd/user/pypoe-web.service`:

```ini
[Unit]
Description=PyPoe Web UI
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/<you>/pypoe
Environment=PYTHONPATH=/home/<you>/pypoe/src
Environment=PATH=/home/<you>/pyenv/bin:/usr/local/bin:/usr/bin:/bin
EnvironmentFile=/home/<you>/pypoe/.env
ExecStart=/bin/bash -lc 'exec /home/<you>/pyenv/bin/pypoe web --host "$(/usr/bin/tailscale ip -4)" --port "${PYPOE_PORT:-8000}"'
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

Save as `~/.config/systemd/user/pypoe-slack.service`:

```ini
[Unit]
Description=PyPoe Slack Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/<you>/pypoe
Environment=PYTHONPATH=/home/<you>/pypoe/src
Environment=PATH=/home/<you>/pyenv/bin:/usr/local/bin:/usr/bin:/bin
EnvironmentFile=/home/<you>/pypoe/.env
ExecStart=/home/<you>/pyenv/bin/pypoe slack
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

Replace `/home/<you>/pypoe` with your clone path and `/home/<you>/pyenv`
with the Python environment that has PyPoe installed (`pip install -e ".[web-ui]"`).
The web unit binds to the Tailscale IP; if you don't use Tailscale,
change `ExecStart` to bind `127.0.0.1` or `0.0.0.0` (with auth — see
[README_WEB.md](README_WEB.md)).

## Enable and start

```bash
systemctl --user daemon-reload
systemctl --user enable --now pypoe-web.service pypoe-slack.service
```

## Manage

```bash
systemctl --user status pypoe-web pypoe-slack
systemctl --user restart pypoe-web pypoe-slack    # after .env or code changes
systemctl --user stop pypoe-web pypoe-slack
journalctl --user -u pypoe-web -u pypoe-slack -f
```

## Run after logout / reboot

User services stop when the user logs out unless lingering is enabled:

```bash
sudo loginctl enable-linger "$USER"
loginctl show-user "$USER" -p Linger    # confirm: Linger=yes
```

## Health checks

```bash
curl -fsS http://<host>:<port>/api/health
systemctl --user is-active pypoe-web pypoe-slack
```

## Notes

- Keep secrets in `.env`; both units load it via `EnvironmentFile=`.
- Prefer binding the web UI to a Tailscale IP over `0.0.0.0`; if you do
  bind to `0.0.0.0`, set `PYPOE_WEB_USERNAME` and `PYPOE_WEB_PASSWORD`.
- Stop `pypoe-slack` before editing Slack bot code so there's only one
  Socket Mode connection at a time, then restart it after.
- The hardened system unit at `scripts/setup/pypoe-web.service` is for a
  dedicated `pypoe` system user; ignore it if you're using the user
  units above.

## Alert investigator

Lab alert investigations use the local `codex exec` CLI with `gpt-5.6-luna`
and `max` reasoning. Configure `lab.alerts.investigation_model` and
`investigation_reasoning_effort` in `src/pypoe/config/slack.yaml`, or override
them with `LAB_INVESTIGATION_MODEL` and `LAB_INVESTIGATION_REASONING_EFFORT`.
The service user must have a working Codex login. `PYPOE_CODEX_BIN` can point
to a custom CLI installation. The CLI must support `--ignore-user-config`,
`--ignore-rules`, and `--ephemeral`.

Investigations load only the lab MCP server, disable shell and app tools,
and run with a read-only sandbox and the existing 300-second timeout.
Runtime state lives in `~/.pypoe/investigator` (override with
`PYPOE_INVESTIGATOR_RUNTIME_DIR`); allow this directory in systemd's
`ReadWritePaths` when filesystem hardening is enabled. Restart the web service
after changing the investigator code or configuration.
