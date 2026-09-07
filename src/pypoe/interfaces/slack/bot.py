"""
PyPoe Slack Bot Integration Module

A comprehensive Slack bot that integrates with Poe API to provide:
- Interactive model selection via Slack UI
- Token/compute point usage monitoring  
- Multi-turn conversations with persistent storage
- Error handling and rate limiting
- Admin controls and usage analytics
- Multiple conversation modes (DM, group, individual)

This module can be imported and used in various ways:
- As a standalone bot: python -m pypoe.slack_bot
- As part of a larger application: from pypoe.slack_bot import PyPoeSlackBot
- Via the CLI: pypoe slack-bot
"""

import asyncio
import os
import re
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict

from ...core.mrkdwn import to_mrkdwn

try:
    from slack_bolt.async_app import AsyncApp
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
    SLACK_IMPORT_ERROR = None
except ImportError as exc:
    SLACK_AVAILABLE = False
    AsyncApp = None
    AsyncSocketModeHandler = None
    SlackApiError = Exception
    SLACK_IMPORT_ERROR = exc

from ...core.client import PoeChatClient
from ...core.history import HistoryManager
from ...core.config import get_config
from ...core.models import (
    CHAT_MODELS,
    DEFAULT_CHAT_MODEL,
    format_model_price_marker,
    get_model_price_markers,
)

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SlackConversationContext:
    """Track Slack-specific conversation context.

    For channels/groups/mpims we scope a conversation to a single Slack
    thread; ``thread_ts`` is the ts of the thread root message. For DMs
    ``thread_ts`` is ``None`` and the context is keyed per-user.
    """
    conversation_id: str
    user_id: str
    channel_id: str
    channel_type: str  # 'im', 'public_channel', 'private_channel', 'group', 'mpim'
    chat_mode: str     # 'slack_dm' or 'slack_thread'
    thread_ts: Optional[str] = None
    owner: Optional[str] = None   # owner-scoping (§4.9): 'slack-dm:<user>' or 'slack-channel:<channel>'
    preferred_model: str = DEFAULT_CHAT_MODEL
    last_activity: datetime = None
    max_context_messages: int = 50  # Default message limit
    max_context_tokens: int = 12000  # Default token limit (conservative)

    def __post_init__(self):
        if self.last_activity is None:
            self.last_activity = datetime.now()
        if self.owner is None:
            # Owner-scoping (§4.9): DMs are per-user; threads are channel-scoped
            # (a thread's audience is the channel members — Slack enforces it).
            self.owner = (
                f"slack-dm:{self.user_id}"
                if self.chat_mode == "slack_dm"
                else f"slack-channel:{self.channel_id}"
            )

class PoeBotUsageTracker:
    """Track usage statistics."""
    
    def __init__(self):
        self.usage_data = {}
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def track_usage(self, user_id: str, model: str, input_text: str, output_text: str):
        """Track usage for a user"""
        if user_id not in self.usage_data:
            self.usage_data[user_id] = {
                "total_messages": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "models_used": {},
                "daily_usage": {},
            }
        
        user_data = self.usage_data[user_id]
        today = datetime.now().strftime("%Y-%m-%d")
        
        input_tokens = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(output_text)
        # Update totals
        user_data["total_messages"] += 1
        user_data["total_input_tokens"] += input_tokens
        user_data["total_output_tokens"] += output_tokens
        
        # Update model usage
        if model not in user_data["models_used"]:
            user_data["models_used"][model] = 0
        user_data["models_used"][model] += 1
        
        # Update daily usage
        if today not in user_data["daily_usage"]:
            user_data["daily_usage"][today] = 0
        user_data["daily_usage"][today] += 1
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get usage statistics for a user"""
        if user_id not in self.usage_data:
            return {
                "total_messages": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "models_used": {},
                "today_usage": 0,
            }
        
        user_data = self.usage_data[user_id]
        today = datetime.now().strftime("%Y-%m-%d")
        today_usage = user_data["daily_usage"].get(today, 0)
        
        return {
            **user_data,
            "today_usage": today_usage,
        }

class PyPoeSlackBot:
    """Main Slack bot class with persistent conversation storage"""
    
    def __init__(self, enable_history: bool = True):
        if not SLACK_AVAILABLE:
            raise ImportError(
                "Slack SDK not available. Install with: pip install slack-bolt slack-sdk"
            )
        
        # Initialize Slack app
        self.app = AsyncApp(
            token=os.environ.get("SLACK_BOT_TOKEN"),
            signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
        )
        
        # Initialize PyPoe client
        self.config = get_config()
        self.hide_thinking_in_slack = self.config.slack_hide_thinking
        self.poe_client = PoeChatClient(enable_history=False)  # We'll handle history ourselves
        
        # Use HistoryManager for persistent storage
        if enable_history:
            from pathlib import Path
            db_path = Path(self.config.database_path)
            self.history = HistoryManager(
                db_path=str(db_path),
                media_dir=str(db_path.parent / "slack_media"),
                enable_media=self.config.enable_media,
            )
        else:
            self.history = None
        
        # Conversation contexts (keyed by conversation_id)
        self.conversation_contexts: Dict[str, SlackConversationContext] = {}
        self.usage_tracker = PoeBotUsageTracker()
        
        # Available models
        self.available_models = []
        
        # Model-specific context limits
        self.model_context_limits = {
            # OpenAI Models
            "GPT-5.4": {"max_tokens": 100000, "max_messages": 200},
            "GPT-4-Turbo": {"max_tokens": 100000, "max_messages": 200},

            # Anthropic Models
            "Claude-Opus-4.8": {"max_tokens": 150000, "max_messages": 300},
            "Claude-Opus-4.7": {"max_tokens": 150000, "max_messages": 300},
            "Claude-Sonnet-4.6": {"max_tokens": 150000, "max_messages": 300},

            # Google Models
            "Gemini-3.1-Pro": {"max_tokens": 800000, "max_messages": 500},
            "Gemini-3-Flash": {"max_tokens": 800000, "max_messages": 500},

            # xAI Models
            "Grok-4": {"max_tokens": 100000, "max_messages": 200},

            # Other Models
            "GLM-5.2": {"max_tokens": 100000, "max_messages": 200},
            "Kimi-K3": {"max_tokens": 100000, "max_messages": 200},

            # Conservative default for anything unlisted
            "Default": {"max_tokens": 12000, "max_messages": 40}
        }
        
        # Set up Slack event handlers
        self._setup_handlers()

        # Optional: /lab-* slash commands. Activated by LAB_API_URL or
        # PYPOE_ENABLE_LAB. Imports are kept local so the slack bot still
        # starts if the lab extra isn't installed.
        self._maybe_register_lab_commands()

    def _maybe_register_lab_commands(self) -> None:
        """Register read-only ``/lab-*`` slash commands if the lab extra is
        installed AND ``LAB_API_URL`` / ``PYPOE_ENABLE_LAB`` is set.

        See ``PyPoe/docs/LAB_INTEGRATION.md`` for the full list. The handlers
        talk to the ``ac-organic-lab`` aggregator over HTTP only — no
        ``/control/*`` calls.
        """
        if not (os.environ.get("LAB_API_URL") or os.environ.get("PYPOE_ENABLE_LAB")):
            return
        try:
            from ...lab.slack_commands import register_lab_commands
            from ...lab.http_client import LabClient
        except ImportError as exc:
            logger.info("Lab slash commands not loaded: %s", exc)
            return

        try:
            self._lab_client = LabClient()
            registered = register_lab_commands(self.app, self._lab_client)
            logger.info(
                "Registered lab slash commands %s against %s",
                registered,
                self._lab_client.base_url,
            )
        except Exception as exc:
            logger.warning("Failed to register lab slash commands: %s", exc)
    
    async def initialize(self):
        """Initialize the bot and fetch available models."""
        try:
            self.available_models = await self.poe_client.get_available_bots()
            if self.history:
                await self.history.initialize()
                # One-shot cleanup of rows produced by the pre-thread-scoping
                # bot (orphan messages keyed by stable Slack ids that no
                # conversation row ever matched, plus the random-UUID
                # conversations created on every /poe invocation).
                if os.environ.get("PYPOE_SLACK_WIPE_ON_START", "").lower() in (
                    "1",
                    "true",
                    "yes",
                ):
                    await self._wipe_slack_history_rows()
            logger.info(
                f"✅ Initialized with {len(self.available_models)} available models"
            )
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {e}")
            self.available_models = list(CHAT_MODELS)

    async def _wipe_slack_history_rows(self) -> None:
        """Delete Slack-originated rows from the history database.

        Runs only when ``PYPOE_SLACK_WIPE_ON_START`` is truthy. Targets:
          * conversations with ``chat_mode LIKE 'slack_%'``
          * any messages still keyed by a ``slack_%`` conversation_id
            (orphans left by the previous broken create-conversation flow)
        """
        if not self.history:
            return
        try:
            import aiosqlite

            async with aiosqlite.connect(self.history.db_path) as db:
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM conversations WHERE chat_mode LIKE 'slack_%'"
                )
                (conv_count,) = await cursor.fetchone()

                cursor = await db.execute(
                    "SELECT COUNT(*) FROM messages "
                    "WHERE conversation_id IN ("
                    "  SELECT id FROM conversations WHERE chat_mode LIKE 'slack_%'"
                    ") OR conversation_id LIKE 'slack_%'"
                )
                (msg_count,) = await cursor.fetchone()

                await db.execute(
                    "DELETE FROM messages "
                    "WHERE conversation_id IN ("
                    "  SELECT id FROM conversations WHERE chat_mode LIKE 'slack_%'"
                    ") OR conversation_id LIKE 'slack_%'"
                )
                await db.execute(
                    "DELETE FROM conversations WHERE chat_mode LIKE 'slack_%'"
                )
                await db.commit()

            logger.info(
                "🧹 PYPOE_SLACK_WIPE_ON_START: removed %d Slack conversations "
                "and %d associated/orphan messages",
                conv_count,
                msg_count,
            )
        except Exception:
            logger.exception("Failed to wipe Slack history rows")
    
    @staticmethod
    def _determine_conversation_strategy(
        channel_type: str,
        user_id: str,
        channel_id: str,
        thread_ts: Optional[str] = None,
    ) -> tuple[str, str, str]:
        """
        Determine conversation strategy based on Slack thread + channel type.

        Scoping (thread takes precedence so threads are always isolated):
          * Any conversation type with ``thread_ts`` set (channel, group,
            mpim, *or DM threads*): per-thread,
            ``slack_thread_{channel_id}_{thread_ts}``.
          * DM (``channel_type == "im"``) without ``thread_ts``: per-user,
            ``slack_dm_{user_id}`` — one persistent conversation per user.
          * Anything else without a thread is invalid (channel/group/mpim
            callers always derive a ``thread_ts``).

        Returns:
            (conversation_id, chat_mode, title)
        """
        if thread_ts:
            conversation_id = f"slack_thread_{channel_id}_{thread_ts}"
            chat_mode = "slack_thread"
            title = f"Slack Thread: {channel_id}/{thread_ts}"
            return conversation_id, chat_mode, title

        if channel_type == "im":
            conversation_id = f"slack_dm_{user_id}"
            chat_mode = "slack_dm"
            title = f"Slack DM: @{user_id}"
            return conversation_id, chat_mode, title

        # Non-DM callers (channel/group/mpim) always derive a thread_ts
        # from the event or the bot's own placeholder post; reaching this
        # branch means a caller forgot to pass one.
        raise ValueError(
            "thread_ts is required for non-DM Slack conversation contexts"
        )

    async def _get_or_create_conversation_context(
        self,
        user_id: str,
        channel_id: str,
        channel_type: str,
        thread_ts: Optional[str] = None,
    ) -> SlackConversationContext:
        """Get or create conversation context with database persistence."""

        conversation_id, chat_mode, title = self._determine_conversation_strategy(
            channel_type, user_id, channel_id, thread_ts
        )

        # Check if context already exists in memory
        if conversation_id in self.conversation_contexts:
            context = self.conversation_contexts[conversation_id]
            context.last_activity = datetime.now()
            return context

        # Ensure the parent conversation row exists so messages aren't orphaned.
        # ``create_conversation`` is idempotent for explicit ids (INSERT OR IGNORE),
        # so calling it on every cold lookup is safe and cheap.
        # Owner-scoping (§4.9): DMs per-user, threads channel-scoped.
        owner = (
            f"slack-dm:{user_id}"
            if chat_mode == "slack_dm"
            else f"slack-channel:{channel_id}"
        )

        if self.history:
            try:
                await self.history.create_conversation(
                    title=title,
                    bot_name=DEFAULT_CHAT_MODEL,
                    chat_mode=chat_mode,
                    conversation_id=conversation_id,
                    owner=owner,
                )
            except Exception as e:
                logger.error(f"Failed to ensure conversation row in database: {e}")

        context = SlackConversationContext(
            conversation_id=conversation_id,
            user_id=user_id,
            channel_id=channel_id,
            channel_type=channel_type,
            chat_mode=chat_mode,
            thread_ts=thread_ts,
            owner=owner,
        )

        # Set appropriate context limits for the default model
        self._update_context_limits_for_model(context)

        self.conversation_contexts[conversation_id] = context
        return context
    
    def _setup_handlers(self):
        """Set up Slack event handlers"""
        
        @self.app.command("/poe")
        async def handle_poe_command(ack, command, respond):
            await ack()
            await self._handle_slash_command(command, respond)
        
        @self.app.event("app_mention")
        async def handle_mentions(event, say):
            await self._handle_mention(event, say)
        
        @self.app.event("message")
        async def handle_dm(event, say):
            # Only respond to DMs, not channel messages
            if event.get("channel_type") == "im":
                await self._handle_direct_message(event, say)
    
    async def _handle_slash_command(self, command, respond):
        """Handle /poe slash commands with conversation context.

        Slash commands don't carry a Slack thread for the bot to reply into,
        so for chat-style invocations in a channel we post a public placeholder
        message ourselves and use *its* ts as the thread root for the
        resulting conversation. Subsequent ``@PyPoe`` mentions in that thread
        will resolve to the same ``slack_thread_<chan>_<ts>`` conversation id.

        Metadata commands (help, models, usage, reset, context, stats,
        set-model) reply ephemerally so they don't pollute the channel.
        """
        user_id = command["user_id"]
        channel_id = command["channel_id"]
        channel_type = command.get("channel_type", "unknown")
        # Slack sometimes includes thread_ts when the slash command is invoked
        # from inside a thread; reuse it when present.
        slash_thread_ts = command.get("thread_ts")
        text = command.get("text", "").strip()

        is_dm = channel_type == "im"

        async def respond_ephemeral(message):
            if isinstance(message, dict):
                payload = {"response_type": "ephemeral", **message}
            else:
                payload = {"response_type": "ephemeral", "text": message}
            # Help / stats / model text is authored in CommonMark; Slack
            # renders mrkdwn. Convert here rather than rewriting every
            # literal (see pypoe.core.mrkdwn).
            if isinstance(payload.get("text"), str):
                payload["text"] = to_mrkdwn(payload["text"])
            await respond(payload)

        async def get_existing_context() -> Optional[SlackConversationContext]:
            """Resolve a context for metadata commands (no new thread root)."""
            if is_dm:
                return await self._get_or_create_conversation_context(
                    user_id, channel_id, channel_type
                )
            if slash_thread_ts:
                return await self._get_or_create_conversation_context(
                    user_id, channel_id, channel_type, thread_ts=slash_thread_ts
                )
            return None

        try:
            if not text or text == "help":
                ctx = await get_existing_context()
                await respond_ephemeral(self._get_help_message_for_slash(ctx, is_dm))
                return

            parts = text.split(" ", 1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if cmd == "models":
                await respond_ephemeral(self._get_models_message())
                return

            if cmd == "usage":
                await respond_ephemeral(self._get_usage_message(user_id))
                return

            if cmd == "set-model":
                if not args:
                    await respond_ephemeral(
                        "❌ Please specify a model. Use `/poe models` to see available options."
                    )
                    return
                ctx = await get_existing_context()
                if ctx is None:
                    await respond_ephemeral(
                        "ℹ️ `/poe set-model` from a channel needs a thread to apply to. "
                        "Run it inside a DM, or run it from inside an existing PyPoe thread."
                    )
                    return
                await self._set_user_model(ctx, args, respond_ephemeral)
                return

            if cmd == "reset":
                ctx = await get_existing_context()
                if ctx is None:
                    await respond_ephemeral(
                        "ℹ️ `/poe reset` from a channel needs a thread to clear. "
                        "Run it inside a DM, or run it from inside the PyPoe thread you "
                        "want to reset. To start fresh in this channel, just run "
                        "`/poe chat …` again — it opens a new thread."
                    )
                    return
                await self._reset_conversation(ctx, respond_ephemeral)
                return

            if cmd == "context":
                ctx = await get_existing_context()
                if ctx is None:
                    await respond_ephemeral(
                        "ℹ️ `/poe context` shows info for a specific conversation. "
                        "Run it inside a DM or inside an existing PyPoe thread."
                    )
                    return
                await respond_ephemeral(self._get_context_info(ctx))
                return

            if cmd == "stats":
                ctx = await get_existing_context()
                if ctx is None:
                    await respond_ephemeral(
                        "ℹ️ `/poe stats` shows stats for a specific conversation. "
                        "Run it inside a DM or inside an existing PyPoe thread."
                    )
                    return
                await respond_ephemeral(await self._get_context_stats(ctx))
                return

            if cmd == "chat":
                if not args:
                    await respond_ephemeral(
                        "❌ Please provide a message. Example: `/poe chat Hello!`"
                    )
                    return

                if is_dm:
                    # DMs: no thread; reply directly in the conversation.
                    ctx = await self._get_or_create_conversation_context(
                        user_id, channel_id, channel_type
                    )
                    await respond_ephemeral(f"📝 You ran `/poe chat {args}`")
                    await self._handle_chat_message(
                        ctx, args, channel_id=channel_id, thread_ts=None
                    )
                    return

                # Channels/groups/mpims: post a public placeholder so its ts can
                # serve as the thread root for this entire conversation.
                if slash_thread_ts:
                    # Already inside a thread (Slack provided thread_ts).
                    placeholder = await self.app.client.chat_postMessage(
                        channel=channel_id,
                        text=(
                            f"🤖 <@{user_id}> asked via `/poe chat`:\n"
                            f"> {args}\n\n_Thinking…_"
                        ),
                        thread_ts=slash_thread_ts,
                    )
                    thread_ts = slash_thread_ts
                else:
                    placeholder = await self.app.client.chat_postMessage(
                        channel=channel_id,
                        text=(
                            f"🤖 <@{user_id}> asked via `/poe chat`:\n"
                            f"> {args}\n\n_Thinking…_"
                        ),
                    )
                    # The placeholder itself is the thread root.
                    thread_ts = placeholder["ts"]
                pending_ts = placeholder["ts"]

                ctx = await self._get_or_create_conversation_context(
                    user_id, channel_id, channel_type, thread_ts=thread_ts
                )
                await self._handle_chat_message(
                    ctx,
                    args,
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    pending_message_ts=pending_ts,
                )
                return

            await respond_ephemeral(
                f"❌ Unknown command: `{cmd}`. Use `/poe help` for available commands."
            )

        except Exception as e:
            logger.error(f"Error handling command: {e}")
            try:
                await respond_ephemeral(f"❌ Error: {str(e)}")
            except Exception:
                logger.exception("Failed to surface slash-command error to user")

    async def _handle_mention(self, event, say):
        """Handle @PyPoe mentions in channels.

        The reply is always placed in the corresponding Slack thread:
          * Mention inside a thread → thread_ts is ``event["thread_ts"]``.
          * Mention at the top level → thread_ts is ``event["ts"]`` (a new
            thread is started rooted at the user's mention).
        """
        user_id = event["user"]
        channel_id = event["channel"]
        channel_type = event.get("channel_type", "public_channel")
        text = event.get("text", "")
        thread_ts = event.get("thread_ts") or event.get("ts")

        # Strip the bot's own mention(s) from the text.
        text = " ".join(
            [word for word in text.split() if not word.startswith("<@")]
        )

        if not text.strip():
            context = await self._get_or_create_conversation_context(
                user_id, channel_id, channel_type, thread_ts=thread_ts
            )
            await self._post_in_thread(
                channel_id, self._get_help_message(context), thread_ts
            )
            return

        context = await self._get_or_create_conversation_context(
            user_id, channel_id, channel_type, thread_ts=thread_ts
        )
        await self._handle_chat_message(
            context, text, channel_id=channel_id, thread_ts=thread_ts
        )

    async def _handle_direct_message(self, event, say):
        """Handle direct messages to the bot.

        DMs without a thread go to one persistent per-user conversation
        (``slack_dm_<user_id>``). If the user replies inside an existing
        thread in the DM, that thread becomes its own per-thread
        conversation and the bot's reply is threaded so it stays in place.
        """
        # Ignore the bot's own messages and message edits/deletes.
        if event.get("bot_id") or event.get("subtype"):
            return

        user_id = event.get("user")
        channel_id = event.get("channel")
        if not user_id or not channel_id:
            # Some message events (e.g. bot/system) don't carry a user id;
            # the previous implementation crashed on event["user"] here.
            return

        text = event.get("text", "")
        # event.thread_ts is set when the message is a reply inside a
        # thread; top-level DM messages have no thread_ts and stay in the
        # persistent per-user conversation.
        thread_ts = event.get("thread_ts")

        if not text.strip():
            context = await self._get_or_create_conversation_context(
                user_id, channel_id, "im", thread_ts=thread_ts
            )
            await self._post_in_thread(
                channel_id, self._get_help_message(context), thread_ts
            )
            return

        context = await self._get_or_create_conversation_context(
            user_id, channel_id, "im", thread_ts=thread_ts
        )
        await self._handle_chat_message(
            context, text, channel_id=channel_id, thread_ts=thread_ts
        )

    async def _post_in_thread(
        self,
        channel_id: str,
        text: str,
        thread_ts: Optional[str],
    ) -> Dict[str, Any]:
        """Post a message via Web API, threading it when ``thread_ts`` is set."""
        kwargs: Dict[str, Any] = {"channel": channel_id, "text": to_mrkdwn(text)}
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        return await self.app.client.chat_postMessage(**kwargs)

    async def _handle_chat_message(
        self,
        context: SlackConversationContext,
        text: str,
        *,
        channel_id: str,
        thread_ts: Optional[str] = None,
        pending_message_ts: Optional[str] = None,
    ):
        """Handle a chat message with persistent conversation history.

        Always posts (or reuses) a placeholder message and then replaces it
        in-place with the model's response, so the Slack thread stays clean
        instead of accumulating "Thinking…" messages.
        """
        try:
            context.last_activity = datetime.now()

            # Reuse a caller-supplied placeholder (slash command path) or post
            # one ourselves so the user sees immediate feedback.
            if pending_message_ts is None:
                placeholder = await self._post_in_thread(
                    channel_id, "🤖 Thinking…", thread_ts
                )
                pending_message_ts = placeholder["ts"]

            # Build the prompt context from persisted history.
            if self.history:
                try:
                    existing_messages = await self.history.get_conversation_messages(
                        context.conversation_id, owner=context.owner
                    )

                    conversation_messages = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in existing_messages
                    ]
                    conversation_messages.append({"role": "user", "content": text})

                    conversation_messages = self._truncate_conversation_context(
                        conversation_messages,
                        context.preferred_model,
                    )

                    # Persist the user message even if it gets truncated out of
                    # the prompt window; full history is what makes /poe stats
                    # and reset behave correctly.
                    await self.history.add_message(
                        conversation_id=context.conversation_id,
                        role="user",
                        content=text,
                        owner=context.owner,
                    )

                except Exception as e:
                    logger.error(f"Failed to load conversation history: {e}")
                    conversation_messages = [{"role": "user", "content": text}]
            else:
                conversation_messages = [{"role": "user", "content": text}]

            full_response = ""
            async for chunk in self.poe_client.send_conversation(
                messages=conversation_messages,
                bot_name=context.preferred_model,
                save_history=False,  # We persist via HistoryManager directly.
            ):
                full_response += chunk

            if self.history and full_response:
                await self.history.add_message(
                    conversation_id=context.conversation_id,
                    role="assistant",
                    content=full_response,
                    bot_name=context.preferred_model,
                    owner=context.owner,
                )

            self.usage_tracker.track_usage(
                context.user_id, context.preferred_model, text, full_response
            )

            response_text = self._format_response_for_slack(
                full_response,
                context.preferred_model,
                context.chat_mode,
            )

            await self.app.client.chat_update(
                channel=channel_id,
                ts=pending_message_ts,
                text=response_text,
            )

        except Exception as e:
            logger.error(f"Error handling chat message: {e}")
            try:
                await self._post_in_thread(
                    channel_id,
                    f"❌ Sorry, I encountered an error: {str(e)}",
                    thread_ts,
                )
            except Exception:
                logger.exception("Failed to surface chat error to Slack")
    
    async def _set_user_model(self, context: SlackConversationContext, model: str, respond_func):
        """Set the preferred model for a conversation context"""
        # Find the closest matching model
        model_lower = model.lower()
        matched_model = None
        
        for available_model in self.available_models:
            if model_lower in available_model.lower():
                matched_model = available_model
                break
        
        if not matched_model:
            available_list = "\n".join([f"• {model}" for model in self.available_models[:10]])
            await respond_func(
                f"❌ Model '{model}' not found.\n\n**Available models:**\n{available_list}\n\n"
                f"Use `/poe models` to see all {len(self.available_models)} available models."
            )
            return
        
        # Set the model
        old_model = context.preferred_model
        context.preferred_model = matched_model
        self._update_context_limits_for_model(context)
        
        await respond_func(
            f"✅ Model changed from **{old_model}** to **{matched_model}**\n"
            f"💰 Price marker: {format_model_price_marker(matched_model)}\n"
            f"📍 Context: {context.chat_mode}"
        )
    
    async def _reset_conversation(self, context: SlackConversationContext, respond_func):
        """Reset the conversation history for a context."""
        try:
            if self.history:
                await self.history.delete_conversation(
                    context.conversation_id, owner=context.owner
                )

                # Re-derive id from the same scoping inputs and re-seed the
                # parent row so future messages aren't orphaned.
                conversation_id, chat_mode, title = self._determine_conversation_strategy(
                    context.channel_type,
                    context.user_id,
                    context.channel_id,
                    thread_ts=context.thread_ts,
                )
                await self.history.create_conversation(
                    title=title,
                    bot_name=context.preferred_model,
                    chat_mode=chat_mode,
                    conversation_id=conversation_id,
                    owner=context.owner,
                )

            await respond_func(f"✅ Conversation reset\n📍 Context: {context.chat_mode}")

        except Exception as e:
            logger.error(f"Error resetting conversation: {e}")
            await respond_func(f"❌ Error resetting conversation: {str(e)}")

    def _get_context_info(self, context: SlackConversationContext) -> str:
        """Get information about the current conversation context."""
        thread_line = (
            f"**Thread Root TS:** `{context.thread_ts}`\n"
            if context.thread_ts
            else ""
        )
        return f"""
📍 **Conversation Context**

**Type:** {context.chat_mode}
**User:** {context.user_id}
**Channel:** {context.channel_id}
{thread_line}**Conversation ID:** `{context.conversation_id}`
**Model:** {context.preferred_model}
**Last Activity:** {context.last_activity.strftime('%Y-%m-%d %H:%M:%S')}

**Context Limits:**
• Max Messages: {context.max_context_messages}
• Max Tokens: {context.max_context_tokens:,}

**Scoping:**
• `slack_dm` → one persistent conversation per user.
• `slack_thread` → one conversation per Slack thread (channel/group/mpim).
"""
    
    async def _get_context_stats(self, context: SlackConversationContext) -> str:
        """Get detailed conversation statistics and context usage"""
        if not self.history:
            return "📊 **Context Stats**\n\nHistory disabled - no statistics available."
        
        try:
            # Get all messages for this conversation
            all_messages = await self.history.get_conversation_messages(
                context.conversation_id, owner=context.owner
            )
            
            if not all_messages:
                return f"""
📊 **Context Stats**

**Conversation:** `{context.conversation_id}`
**Model:** {context.preferred_model}

**Message Count:** 0
**Status:** New conversation

**Model Limits:**
• Max Messages: {context.max_context_messages}
• Max Tokens: {context.max_context_tokens:,}
"""
            
            # Convert to API format for token estimation
            api_messages = []
            for msg in all_messages:
                api_messages.append({
                    'role': msg['role'], 
                    'content': msg['content']
                })
            
            # Estimate tokens for full conversation
            total_tokens = sum(self._estimate_message_tokens(msg) for msg in api_messages)
            
            # See what would be included with current limits
            truncated_messages = self._truncate_conversation_context(api_messages, context.preferred_model)
            active_tokens = sum(self._estimate_message_tokens(msg) for msg in truncated_messages)
            
            # Count message types
            user_messages = len([m for m in all_messages if m['role'] == 'user'])
            assistant_messages = len([m for m in all_messages if m['role'] == 'assistant'])
            
            # Calculate percentages
            token_usage_pct = (active_tokens / context.max_context_tokens) * 100 if context.max_context_tokens > 0 else 0
            message_usage_pct = (len(truncated_messages) / context.max_context_messages) * 100 if context.max_context_messages > 0 else 0
            
            truncation_info = ""
            if len(truncated_messages) < len(all_messages):
                truncated_count = len(all_messages) - len(truncated_messages)
                truncation_info = f"\n⚠️ **{truncated_count} messages truncated** from context"
            
            return f"""
📊 **Context Stats**

**Conversation:** `{context.conversation_id}`
**Model:** {context.preferred_model}

**Total Messages:** {len(all_messages)} ({user_messages} user, {assistant_messages} assistant)
**Active Context:** {len(truncated_messages)} messages
**Total Tokens:** ≈{total_tokens:,}
**Active Tokens:** ≈{active_tokens:,} ({token_usage_pct:.1f}% of limit)

**Model Limits:**
• Max Messages: {context.max_context_messages} ({message_usage_pct:.1f}% used)
• Max Tokens: {context.max_context_tokens:,} ({token_usage_pct:.1f}% used)

**Status:** {"🟢 Within limits" if len(truncated_messages) == len(all_messages) else "🟡 Context truncated"}{truncation_info}

💡 Use `/poe reset` to clear history if context becomes too large
"""
            
        except Exception as e:
            logger.error(f"Error getting context stats: {e}")
            return f"❌ Error retrieving context statistics: {str(e)}"
    
    def _get_help_message(self, context: SlackConversationContext) -> str:
        """Get help message scoped to a specific conversation context."""
        return f"""
🤖 **PyPoe Slack Bot - Help**

**Current Context:** {context.chat_mode}
**Your Model:** {context.preferred_model}

**Slash Commands:**
• `/poe help` - Show this help
• `/poe models` - List available AI models
• `/poe chat <message>` - Send a message (opens a thread in channels)
• `/poe set-model <model>` - Set the model for this conversation
• `/poe usage` - Check your token usage stats
• `/poe reset` - Reset this conversation's history
• `/poe context` - Show conversation context info
• `/poe stats` - Show detailed context statistics

**Direct Interaction:**
• `@PyPoe <message>` - Mention the bot to start a thread, or reply in
  an existing thread to continue that conversation.
• DM the bot directly for a private, persistent conversation.

**How conversations are scoped:**
• 💬 DMs → one persistent conversation per user.
• 🧵 Channels / groups → one conversation per Slack thread. Replies in
  the thread (with `@PyPoe`) continue the same context. Start a new
  thread to start a new conversation.

**Current Status:** ✅ Connected to Poe API
"""

    def _get_help_message_for_slash(
        self,
        context: Optional[SlackConversationContext],
        is_dm: bool,
    ) -> str:
        """Help variant for slash commands invoked outside any thread."""
        if context is not None:
            return self._get_help_message(context)

        scope_hint = (
            "You ran `/poe help` from a channel without an active thread, "
            "so per-thread commands (`reset`, `context`, `stats`, `set-model`) "
            "have nothing to act on yet. Use `/poe chat …` to open a thread, "
            "or run those commands from a DM."
        )
        return f"""
🤖 **PyPoe Slack Bot - Help**

{scope_hint}

**Slash Commands:**
• `/poe help` - Show this help
• `/poe models` - List available AI models
• `/poe chat <message>` - Send a message (opens a thread in channels)
• `/poe set-model <model>` - Set the model for this conversation
• `/poe usage` - Check your token usage stats
• `/poe reset` - Reset this conversation's history
• `/poe context` - Show conversation context info
• `/poe stats` - Show detailed context statistics

**Direct Interaction:**
• `@PyPoe <message>` - Mention the bot to start a thread, or reply in
  an existing thread to continue that conversation.
• DM the bot directly for a private, persistent conversation.

**How conversations are scoped:**
• 💬 DMs → one persistent conversation per user.
• 🧵 Channels / groups → one conversation per Slack thread.
"""
    
    def _get_models_message(self) -> str:
        """Get available models message"""
        if not self.available_models:
            return "❌ No models available. Please check the bot configuration."

        def format_price_row(label: str, marker: str, width: int = 48) -> str:
            """Wrap long price markers with continuation aligned under '$'."""
            prefix = "    In:  " if label == "In" else "    Out: "
            continuation_prefix = " " * len(prefix)
            chunks = [marker[i:i + width] for i in range(0, len(marker), width)] or ["-"]
            lines = [f"{prefix}{chunks[0]}"]
            lines.extend(f"{continuation_prefix}{chunk}" for chunk in chunks[1:])
            return "\n".join(lines)
        
        # Group models by provider
        providers = {}
        for model in self.available_models:
            if "GPT" in model or "gpt" in model:
                provider = "OpenAI"
            elif "Claude" in model:
                provider = "Anthropic"
            elif "Gemini" in model or "PaLM" in model:
                provider = "Google"
            elif "Llama" in model:
                provider = "Meta"
            else:
                provider = "Other"
            
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(model)
        
        message = f"🤖 **Available AI Models ({len(self.available_models)} total)**\n\n"
        
        for provider, models in providers.items():
            message += f"**{provider}:**\n"
            message += "```\n"
            for model in models[:5]:  # Limit to first 5 per provider
                input_marker, output_marker = get_model_price_markers(model)
                message += f"{model}\n"
                message += format_price_row("In", input_marker) + "\n"
                message += format_price_row("Out", output_marker) + "\n"
            message += "```\n"
            if len(models) > 5:
                message += f"• ... and {len(models) - 5} more\n"
            message += "\n"
        
        message += (
            "💡 Price markers use one `$` per $1.00 per 1M tokens "
            "(In/Out).\n"
            "💡 Use `/poe set-model <model-name>` to switch models"
        )
        return message
    
    def _get_usage_message(self, user_id: str) -> str:
        """Get usage statistics message"""
        stats = self.usage_tracker.get_user_stats(user_id)
        
        if stats["total_messages"] == 0:
            return "📊 **Your Usage Stats**\n\nNo messages sent yet. Try `/poe chat Hello!`"
        
        # Format top models
        top_models = sorted(
            stats["models_used"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        top_models_text = "\n".join([
            f"• {model}: {count} messages" 
            for model, count in top_models
        ])
        
        return f"""
📊 **Your Usage Stats**

**Total Activity:**
• Messages sent: {stats['total_messages']:,}
• Estimated input tokens: {stats['total_input_tokens']:,}
• Estimated output tokens: {stats['total_output_tokens']:,}
**Today's Usage:**
• Messages: {stats['today_usage']}

**Top Models Used:**
{top_models_text}

💡 *Each conversation context maintains separate history*
"""
    
    @staticmethod
    def _hide_thinking_for_slack(response: str) -> str:
        """Remove model reasoning blocks from Slack-visible output."""
        thinking_block_pattern = re.compile(
            r"<(think|thinking|reasoning)\b[^>]*>.*?</\1>",
            re.IGNORECASE | re.DOTALL,
        )
        redacted = thinking_block_pattern.sub("", response)
        markdown_thinking_pattern = re.compile(
            r"^\s*(?:\*\*)?thinking(?:\.\.\.|…)?(?:\*\*)?\s*\n+"
            r"(?:>[^\n]*(?:\n|$))+",
            re.IGNORECASE,
        )
        redacted = markdown_thinking_pattern.sub("", redacted)
        redacted = re.sub(r"\n{3,}", "\n\n", redacted).strip()
        return redacted or "_(thinking hidden)_"

    def _format_response_for_slack(self, response: str, model: str, chat_mode: str) -> str:
        """Format the AI response for Slack with context info"""
        if getattr(self, "hide_thinking_in_slack", False):
            response = self._hide_thinking_for_slack(response)

        # Truncate very long responses
        if len(response) > 3000:
            response = response[:2950] + "\n\n... *(response truncated)*"
        
        context_indicator = {
            "slack_dm": "🔒 DM",
            "slack_thread": "🧵 Thread",
        }.get(chat_mode, "❓ Unknown")

        # Model replies are markdown, and the header below is written in it
        # too. One conversion covers both (see pypoe.core.mrkdwn).
        return to_mrkdwn(f"🤖 **{model}** {context_indicator}\n\n{response}")
    
    def _estimate_message_tokens(self, message: Dict[str, str]) -> int:
        """Estimate tokens for a message (rough approximation)"""
        content = message.get('content', '')
        role = message.get('role', '')
        
        # Rough estimation: 1 token ≈ 4 characters for text
        # Add overhead for role, formatting, etc.
        base_tokens = len(content) // 4
        overhead_tokens = 10  # Role, formatting overhead
        
        return base_tokens + overhead_tokens
    
    def _get_model_limits(self, model_name: str) -> Dict[str, int]:
        """Get context limits for a specific model"""
        # Try exact match first
        if model_name in self.model_context_limits:
            return self.model_context_limits[model_name]
        
        # Try partial matches for model families
        for known_model, limits in self.model_context_limits.items():
            if known_model != "Default" and any(
                part in model_name for part in known_model.split("-")[:2]
            ):
                return limits
        
        # Fallback to conservative defaults
        return self.model_context_limits["Default"]
    
    def _truncate_conversation_context(self, messages: List[Dict[str, str]], model_name: str) -> List[Dict[str, str]]:
        """
        Intelligently truncate conversation context to fit model limits.
        
        Strategy:
        1. Always keep the most recent messages
        2. Try to preserve conversation flow
        3. Keep important context (user questions, model switches)
        4. Respect both token and message count limits
        """
        if not messages:
            return messages
        
        limits = self._get_model_limits(model_name)
        max_tokens = limits["max_tokens"]
        max_messages = limits["max_messages"]
        
        # If we're within limits, return as-is
        if len(messages) <= max_messages:
            total_tokens = sum(self._estimate_message_tokens(msg) for msg in messages)
            if total_tokens <= max_tokens:
                return messages
        
        # Need to truncate - use sliding window approach
        # Always keep the most recent messages, working backwards
        truncated_messages = []
        total_tokens = 0
        
        # Start from the end (most recent) and work backwards
        for message in reversed(messages):
            message_tokens = self._estimate_message_tokens(message)
            
            # Check if adding this message would exceed limits
            if (len(truncated_messages) >= max_messages or 
                total_tokens + message_tokens > max_tokens):
                break
            
            truncated_messages.insert(0, message)  # Insert at beginning
            total_tokens += message_tokens
        
        # Ensure we have at least some context
        if not truncated_messages and messages:
            # If even the most recent message exceeds limits, take it anyway
            # The API will handle the overflow
            truncated_messages = messages[-1:]
        
        # Log truncation for debugging
        if len(truncated_messages) < len(messages):
            logger.info(
                f"Context truncated for {model_name}: "
                f"{len(messages)} → {len(truncated_messages)} messages "
                f"(≈{total_tokens} tokens)"
            )
        
        return truncated_messages
    
    def _update_context_limits_for_model(self, context: SlackConversationContext):
        """Update context limits when model changes"""
        limits = self._get_model_limits(context.preferred_model)
        context.max_context_tokens = limits["max_tokens"]
        context.max_context_messages = limits["max_messages"]
    
    async def run(self):
        """Run the Slack bot"""
        await self.initialize()
        
        # Use Socket Mode for local development
        if os.environ.get("SLACK_SOCKET_MODE", "true").lower() == "true":
            handler = AsyncSocketModeHandler(self.app, os.environ.get("SLACK_APP_TOKEN"))
            await handler.start_async()
        else:
            # Use HTTP mode for production
            await self.app.async_start(port=int(os.environ.get("PORT", 3000)))
    
    async def close(self):
        """Clean up resources"""
        await self.poe_client.close()
        if self.history:
            await self.history.close()

async def main():
    """Main entry point"""
    import sys

    # Load the same .env candidates used by the web/CLI config before checking
    # Slack-specific environment variables.
    try:
        get_config()
    except ValueError:
        # get_config() still loads .env before validating POE_API_KEY. Keep
        # going so the required-variable report can show all missing values.
        pass

    # Check required environment variables
    required_vars = ["SLACK_BOT_TOKEN", "SLACK_SIGNING_SECRET", "POE_API_KEY"]
    if os.environ.get("SLACK_SOCKET_MODE", "true").lower() == "true":
        required_vars.append("SLACK_APP_TOKEN")
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("\n📋 Setup Instructions:")
        print("1. Create a Slack app at https://api.slack.com/apps")
        print("2. Set environment variables:")
        print("   export SLACK_BOT_TOKEN=xoxb-your-bot-token")
        print("   export SLACK_SIGNING_SECRET=your-signing-secret")
        print("   export SLACK_APP_TOKEN=xapp-your-app-token  # For Socket Mode")
        print("   export POE_API_KEY=your-poe-api-key")
        print("3. Run: pypoe slack")
        return
    
    if not SLACK_AVAILABLE:
        print("❌ Slack SDK not installed. Install with:")
        print("   pip install slack-bolt slack-sdk aiohttp")
        if SLACK_IMPORT_ERROR:
            print(f"   Import error: {SLACK_IMPORT_ERROR}")
        return
    
    print("🚀 Starting PyPoe Slack Bot...")
    print("📋 Configuration:")
    print(f"   POE_API_KEY: {'✅ Set' if os.environ.get('POE_API_KEY') else '❌ Missing'}")
    print(f"   SLACK_BOT_TOKEN: {'✅ Set' if os.environ.get('SLACK_BOT_TOKEN') else '❌ Missing'}")
    print(f"   Socket Mode: {os.environ.get('SLACK_SOCKET_MODE', 'true')}")
    print("   Database: HistoryManager with media support")
    print("   Conversation Strategy: Individual contexts per user")
    print("   Context Management: Intelligent truncation with model-specific limits")
    
    bot = PyPoeSlackBot()
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        print("\n👋 Shutting down PyPoe Slack Bot...")
        await bot.close()
    except Exception as e:
        print(f"❌ Error running bot: {e}")
        await bot.close()

if __name__ == "__main__":
    asyncio.run(main()) 