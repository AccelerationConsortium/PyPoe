"""Read-only + journaling + consultation MCP server for the AC Organic
Self-driving Lab.

Runs over stdio so Claude Desktop / Claude Code can register it with::

    claude mcp add ac-organic-lab -- pypoe lab-mcp

Tools fall into three buckets:

* **Read** — wrap the aggregator's HTTP API. Source of truth.
* **Write (journaling only)** — ``append_observation`` posts to the
  aggregator's ``/api/ingest/events`` so an agent's findings show up in
  the dashboard's history sidebar. **No control endpoints.**
* **Other** — ``consult_poe`` (second opinion via PyPoe's own CLI) and
  ``ask_human`` (Slack thread reply with polling).

Explicitly NOT exposed: ``control_action``. Direct control would bypass
Layer 3 / Layer 4 interlocks per
``ac-organic-lab/docs/INTERLOCKS.md``. Control belongs to the
``lab-skills`` SDK; when v0.4 ships, register both
``pypoe lab-mcp`` and ``lab-skills mcp serve``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - lab extra not installed
    FastMCP = None  # type: ignore[assignment]
    _MCP_IMPORT_ERROR = exc
else:
    _MCP_IMPORT_ERROR = None

from ..core.mrkdwn import to_mrkdwn
from .http_client import LabClient

logger = logging.getLogger(__name__)


def build_server(client: Optional[LabClient] = None) -> "FastMCP":
    """Build (but don't run) the FastMCP server.

    Pulled out as a separate function so tests can introspect the tool
    list without running the stdio loop.
    """
    if FastMCP is None:
        raise ImportError(
            "mcp package is required for pypoe lab-mcp — install the "
            "lab extra: pip install -e '.[lab]'"
        ) from _MCP_IMPORT_ERROR

    server = FastMCP("ac-organic-lab")
    lab = client or LabClient()

    # ------------------------------------------------------------------ read

    @server.tool()
    async def list_equipment() -> dict:
        """Return every registered device with its latest status."""
        return await lab.list_equipment()

    @server.tool()
    async def get_equipment_status(equipment_id: str) -> dict:
        """Return one device's full STATUS_SPEC envelope.

        Inspect ``status.equipment_status``, ``status.message``,
        ``status.last_error``, ``allowed_actions`` (STATUS_SPEC v1.1),
        and ``details.claimed_by`` for who currently holds the device.
        """
        return await lab.get_equipment_status(equipment_id)

    @server.tool()
    async def aggregator_health() -> dict:
        """Aggregator service health + equipment count."""
        return await lab.health()

    @server.tool()
    async def list_platforms() -> dict:
        """Return the dashboard's section layout (Overview groupings)."""
        return await lab.platforms()

    @server.tool()
    async def skill_catalog() -> dict:
        """Static catalog of every skill the SDK can dispatch, by platform.

        Read-only. Calling a skill goes through ``lab-skills``, not here.
        """
        return await lab.catalog()

    @server.tool()
    async def recent_events(device_id: str, limit: int = 50) -> dict:
        """State transitions, errors, startup/shutdown events for one device."""
        return await lab.recent_events(device_id, limit=limit)

    @server.tool()
    async def recent_observations(device_id: str, limit: int = 10) -> dict:
        """Prior agent findings for one device, newest first.

        These are earlier investigations' own notes (the rows this server
        wrote via ``append_observation``). Read them BEFORE diagnosing so you
        can tell a fresh incident from a recurrence and build on the last
        root cause instead of starting from scratch.
        """
        return await lab.recent_observations(device_id, limit=limit)

    @server.tool()
    async def device_uptime(
        device_id: Optional[str] = None, days: int = 7
    ) -> dict:
        """Uptime % over the last N days. Omit device_id for the whole fleet."""
        return await lab.uptime(device_id=device_id, days=days)

    @server.tool()
    async def latest_sensors() -> dict:
        """Most recent reading per (sensor_id, metric) across the fleet."""
        return await lab.latest_sensors()

    @server.tool()
    async def recent_runs(limit: int = 20) -> dict:
        """Most recent dosing runs, newest first."""
        return await lab.recent_runs(limit=limit)

    @server.tool()
    async def run_wells(run_id: str) -> dict:
        """Per-well dispense results for one run (96 rows for a full plate)."""
        return await lab.run_wells(run_id)

    # ------------------------------------------------------------------ write

    @server.tool()
    async def append_observation(
        device_id: str,
        summary: str,
        severity: str = "info",
        extra: Optional[dict] = None,
    ) -> dict:
        """Journal an agent observation to the aggregator's history.

        Surfaces in ``GET /api/history/events/{device_id}`` and on the
        dashboard's history sidebar. ``severity`` is one of
        ``info``/``warning``/``error``.
        """
        return await lab.append_observation(
            device_id, summary, severity=severity, extra=extra
        )

    # ------------------------------------------------------------------ other LLM

    @server.tool()
    async def consult_poe(
        model: str, question: str, context: Optional[str] = None
    ) -> dict:
        """Ask another model (via Poe) for a second opinion.

        Shells out to ``pypoe cli chat`` with ``--bot <model>``. Costs
        Poe compute points. Use sparingly — when Claude's confidence in
        a diagnosis is low, or for cross-checking on ambiguous failures.
        """
        return await _consult_poe(model, question, context)

    @server.tool()
    async def ask_human(
        question: str,
        channel: Optional[str] = None,
        timeout_s: int = 600,
    ) -> dict:
        """Post a question to Slack and wait for the first non-bot reply.

        Returns ``{"answer": str | None, "timed_out": bool, "thread_ts": str}``.
        The thread stays open after the timeout, so humans can still
        answer later — the answer just won't be visible to *this* call.
        """
        return await _ask_human(question, channel=channel, timeout_s=timeout_s)

    return server


# ---------------------------------------------------------------------------
# Tool implementations broken out so tests can mock them.
# ---------------------------------------------------------------------------


async def _consult_poe(
    model: str, question: str, context: Optional[str]
) -> dict:
    """Ask another Poe model for a second opinion via PoeChatClient.

    Uses ``pypoe.core.client.PoeChatClient`` directly rather than shelling
    out to ``pypoe cli chat``: no subprocess, no shell-escaping pitfalls,
    no spurious "CLI Chat …" rows in PyPoe's history DB. The consult is
    ephemeral by design — passing ``save_history=False`` means nothing
    persists between calls.

    Return shape matches what the MCP tool surface promises:
    ``{model, answer, stderr, returncode}`` where ``returncode`` is 0
    on success and non-zero on Poe-side errors / config problems.
    """
    prompt = f"{context}\n\n{question}" if context else question

    try:
        from ..core.client import PoeChatClient
    except ImportError as exc:  # pragma: no cover - pypoe install broken
        return {
            "model": model,
            "answer": "",
            "stderr": f"PoeChatClient unavailable: {exc}",
            "returncode": 127,
        }

    try:
        client = PoeChatClient(enable_history=False)
        chunks: list[str] = []
        async for chunk in client.send_message(
            prompt, bot_name=model, save_history=False
        ):
            chunks.append(chunk)
        return {
            "model": model,
            "answer": "".join(chunks).strip(),
            "stderr": "",
            "returncode": 0,
        }
    except ValueError as exc:
        # PoeChatClient raises ValueError for unknown / inaccessible bots,
        # missing API key, quota exceeded etc. — these are caller-fixable.
        return {
            "model": model,
            "answer": "",
            "stderr": str(exc),
            "returncode": 2,
        }
    except Exception as exc:  # pragma: no cover - network / Poe API blip
        return {
            "model": model,
            "answer": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "returncode": 1,
        }


async def _ask_human(
    question: str,
    *,
    channel: Optional[str] = None,
    timeout_s: int = 600,
    poll_interval_s: float = 5.0,
) -> dict:
    """Post a question to Slack and poll for the first non-bot reply."""
    try:
        from slack_sdk.web.async_client import AsyncWebClient
    except ImportError as exc:
        return {
            "answer": None,
            "timed_out": False,
            "error": f"slack_sdk not installed: {exc}",
        }

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        return {"answer": None, "timed_out": False, "error": "SLACK_BOT_TOKEN unset"}

    from .config import load_config
    ch = channel or load_config().slack.alert_channel
    slack = AsyncWebClient(token=token)

    # ``question`` is model-authored, so it arrives as markdown; Slack
    # renders mrkdwn (see pypoe.core.mrkdwn).
    post = await slack.chat_postMessage(
        channel=ch, text=to_mrkdwn(f":question: {question}")
    )
    thread_ts = post["ts"]
    channel_id = post["channel"]

    deadline = time.monotonic() + timeout_s
    bot_user_id: Optional[str] = None
    try:
        auth = await slack.auth_test()
        bot_user_id = auth.get("user_id")
    except Exception:
        bot_user_id = None

    while time.monotonic() < deadline:
        await asyncio.sleep(poll_interval_s)
        replies = await slack.conversations_replies(
            channel=channel_id, ts=thread_ts
        )
        for msg in replies.get("messages", []):
            if msg.get("ts") == thread_ts:
                continue  # skip our own prompt
            if msg.get("bot_id"):
                continue
            if bot_user_id and msg.get("user") == bot_user_id:
                continue
            return {
                "answer": msg.get("text", ""),
                "timed_out": False,
                "thread_ts": thread_ts,
                "channel": channel_id,
                "user": msg.get("user"),
            }

    return {
        "answer": None,
        "timed_out": True,
        "thread_ts": thread_ts,
        "channel": channel_id,
    }


# ---------------------------------------------------------------------------
# Entry point used by `pypoe lab-mcp`.
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server on stdio until the parent process closes the pipe."""
    logging.basicConfig(
        level=os.environ.get("LAB_MCP_LOG_LEVEL", "WARNING"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    server = build_server()
    server.run()


if __name__ == "__main__":  # pragma: no cover
    main()
