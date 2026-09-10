import asyncio
import re
from pathlib import Path
from typing import AsyncGenerator, List, Dict, Any, Optional
import fastapi_poe as fp

from .config import get_config, Config
from .models import CHAT_MODELS, DEFAULT_CHAT_MODEL, provider_for, refresh_openrouter_catalog
from .provider_health import (
    ACCOUNT_BLOCKING_REASONS,
    MODEL_SCOPED_REASONS,
    classify_exception,
    recovery_hint,
    registry as health,
)
from .providers import (
    OPENROUTER,
    ProviderSpec,
    api_key_for,
    get_provider,
    stream_openai_compatible,
)

try:
    from .history import HistoryManager, _UNSCOPED
    HISTORY_AVAILABLE = True
except ImportError:
    HISTORY_AVAILABLE = False
    HistoryManager = None
    _UNSCOPED = None

class ContentProcessor:
    """Utility class for processing and filtering API responses."""
    
    def __init__(self):
        self.last_generating_message = ""
        self.generating_pattern = re.compile(r'^Generating\.+(\s*\(\d+s elapsed\))?$')
        self.image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
        self.video_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]*\.(?:mp4|mov|avi|webm|mkv|flv)[^)]*)\)', re.IGNORECASE)
        
    def should_filter_chunk(self, text: str) -> bool:
        """Determine if a text chunk should be filtered out."""
        if not text or not text.strip():
            return True
            
        # Handle generating/thinking messages
        if self.generating_pattern.match(text.strip()):
            if text.strip() == self.last_generating_message:
                return True  # Skip duplicate
            self.last_generating_message = text.strip()
            return False  # Allow first generating message to show (important for image generation)
            
        # Reset generating state when real content arrives
        if self.last_generating_message:
            self.last_generating_message = ""
            
        return False
    
    def process_content_for_display(self, content: str) -> str:
        """Process content for display, converting images and videos to inline elements."""
        if not content:
            return content
        
        processed = content
        
        # Convert videos first (more specific pattern)
        def replace_video(match):
            alt_text = match.group(1) or "Generated Video"
            url = match.group(2)
            return f'<video controls style="max-width: 100%; height: auto; border-radius: 8px; margin: 8px 0; display: block;" poster="" preload="metadata"><source src="{url}" type="video/mp4">Your browser does not support the video tag. <a href="{url}" target="_blank" class="video-fallback-link" style="color: #3498db; text-decoration: none;">{alt_text} (Click to open)</a></video>'
        
        processed = self.video_pattern.sub(replace_video, processed)
        
        # Convert images (excluding videos that were already processed)
        def replace_image(match):
            alt_text = match.group(1) or "Generated Image"
            url = match.group(2)
            # Skip if this looks like a video URL that should have been caught by video pattern
            if any(ext in url.lower() for ext in ['.mp4', '.mov', '.avi', '.webm', '.mkv', '.flv']):
                return match.group(0)  # Return original text
            return f'<img src="{url}" alt="{alt_text}" style="max-width: 100%; height: auto; border-radius: 8px; margin: 8px 0; display: block;" loading="lazy" onerror="this.style.display=\'none\'; this.nextElementSibling.style.display=\'inline-block\';" /><a href="{url}" target="_blank" class="image-fallback-link" style="display: none; color: #3498db; text-decoration: none;">{alt_text} (Click to open)</a>'
        
        processed = self.image_pattern.sub(replace_image, processed)
        return processed
    
    def extract_media_urls(self, content: str) -> List[Dict[str, str]]:
        """Extract media URLs from content for storage tracking."""
        media_urls = []
        
        # Extract videos first (more specific pattern)
        for match in self.video_pattern.finditer(content):
            alt_text = match.group(1) or "Generated Video"
            url = match.group(2)
            media_urls.append({
                'type': 'video',
                'url': url,
                'alt_text': alt_text,
                'filename': self._extract_filename_from_url(url)
            })
        
        # Extract images (excluding videos that were already processed)
        for match in self.image_pattern.finditer(content):
            alt_text = match.group(1) or "Generated Image"
            url = match.group(2)
            # Skip if this looks like a video URL that should have been caught by video pattern
            if any(ext in url.lower() for ext in ['.mp4', '.mov', '.avi', '.webm', '.mkv', '.flv']):
                continue  # Skip this URL
            media_urls.append({
                'type': 'image',
                'url': url,
                'alt_text': alt_text,
                'filename': self._extract_filename_from_url(url)
            })
        
        return media_urls
    
    def _extract_filename_from_url(self, url: str) -> str:
        """Extract a filename from a URL for storage purposes."""
        # Try to extract filename from URL
        parts = url.split('/')
        if parts:
            filename = parts[-1].split('?')[0]  # Remove query parameters
            if '.' in filename:
                return filename
        
        # Generate a fallback filename based on URL content
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        # Check if this looks like a video URL
        video_extensions = ['.mp4', '.mov', '.avi', '.webm', '.mkv', '.flv']
        if any(ext in url.lower() for ext in video_extensions):
            return f"video_{url_hash}.mp4"
        else:
            return f"image_{url_hash}.png"

class PoeChatClient:
    """A high-level client for interacting with Poe.com using the official API."""

    def __init__(self, config: Config = None, enable_history: bool = True,
                 owner: Optional[str] = None):
        if config is None:
            config = get_config()
        
        self.config = config
        self.api_key = config.poe_api_key
        self.enable_history = enable_history and HISTORY_AVAILABLE
        self.content_processor = ContentProcessor()

        # Owner-scoping (CLAUDE.local.md §4.9): when set (e.g. the CLI's OS
        # user), every history op this client performs is stamped/scoped to
        # this principal. ``None`` = unscoped (the transition default; the web
        # UI stays unscoped until it is gated per §4.8).
        self._owner = owner
        
        if self.enable_history:
            # Setup media directory for history (only materialized if media
            # auto-download is enabled).
            media_dir = Path(self.config.database_path).parent / "media"

            self.history = HistoryManager(
                db_path=str(self.config.database_path),
                media_dir=str(media_dir),
                enable_media=self.config.enable_media,
                default_owner=(owner if owner is not None else _UNSCOPED),
            )
            self._history_initialized = False
        else:
            self.history = None
            self._history_initialized = True

    async def _ensure_history_initialized(self):
        """Ensure the history database is initialized."""
        if self.enable_history and not self._history_initialized:
            await self.history.initialize()
            self._history_initialized = True

    def _convert_role_for_api(self, role: str) -> str:
        """Convert role names for API compatibility."""
        if role == "assistant":
            return "bot"
        return role

    def _convert_role_for_history(self, role: str) -> str:
        """Convert role names for history storage."""
        if role == "bot":
            return "assistant"
        return role

    async def send_message(
        self, 
        message: str, 
        bot_name: str = DEFAULT_CHAT_MODEL,
        conversation_id: Optional[str] = None,
        save_history: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Send a message to a Poe bot and stream the response.
        
        If conversation_id is provided, automatically retrieves and includes
        conversation history to maintain context continuity.
        
        Args:
            message: The message to send
            bot_name: The bot to send the message to
            conversation_id: Optional conversation ID for history tracking
            save_history: Whether to save the conversation to history
            
        Yields:
            Partial responses from the bot
            
        Raises:
            ValueError: If the bot is not accessible or available
            Exception: For other API errors
        """
        await self._ensure_history_initialized()
        
        # If conversation_id is provided and history is enabled, include conversation context
        if conversation_id and self.enable_history:
            try:
                # Retrieve existing conversation history
                existing_messages = await self.get_conversation_messages(conversation_id)
                
                # Convert to the format expected by send_conversation
                conversation_messages = []
                for msg in existing_messages:
                    conversation_messages.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })
                
                # Add the new user message
                conversation_messages.append({
                    'role': 'user',
                    'content': message
                })
                
                # Save the new user message to history manually
                if save_history:
                    await self.history.add_message(
                        conversation_id=conversation_id,
                        role="user",
                        content=message,
                        bot_name=bot_name
                    )
                
                # Use send_conversation for full context but don't save history 
                # (to avoid duplicating existing messages)
                full_response = ""
                async for partial in self.send_conversation(
                    messages=conversation_messages,
                    bot_name=bot_name,
                    conversation_id=conversation_id,
                    save_history=False  # Prevent duplicate history entries
                ):
                    full_response += partial
                    yield partial
                
                # Save the bot response to history manually
                if save_history and full_response:
                    await self.history.add_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=full_response,
                        bot_name=bot_name
                    )
                return
                
            except Exception as e:
                # If conversation history retrieval fails, fall back to single message
                # This ensures backward compatibility
                print(f"Warning: Failed to retrieve conversation history: {e}")
                print("Falling back to single message mode...")
        
        # Original single-message logic for new conversations or when history is disabled
        # Create a new conversation if none provided and history is enabled
        if conversation_id is None and save_history and self.enable_history:
            conversation_id = await self.history.create_conversation(
                title=f"Chat with {bot_name}",
                bot_name=bot_name,
                chat_mode="chatbot",
                topic=None  # Topic will be generated from first message if needed
            )
        
        # Save the user message to history
        if save_history and conversation_id and self.enable_history:
            await self.history.add_message(
                conversation_id=conversation_id,
                role="user",
                content=message,
                bot_name=bot_name
            )
        
        # Reset content processor state for new request
        self.content_processor.last_generating_message = ""

        # Stream the response with error handling and content filtering
        full_response = ""
        try:
            async for chunk in self._stream(bot_name, [{"role": "user", "content": message}]):
                yield chunk
                full_response += chunk
        except Exception as e:
            translated = await self._provider_error(e, bot_name)
            if translated is e:
                raise
            raise translated from e

        # Save the bot response to history
        if save_history and conversation_id and full_response and self.enable_history:
            await self.history.add_message(
                conversation_id=conversation_id,
                role="assistant",  # Save as assistant in history
                content=full_response
            )

    async def send_conversation(
        self, 
        messages: List[Dict[str, str]], 
        bot_name: str = DEFAULT_CHAT_MODEL,
        conversation_id: Optional[str] = None,
        save_history: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Send a multi-turn conversation to a Poe bot.
        
        Args:
            messages: List of messages in format [{"role": "user", "content": "..."}, ...]
            bot_name: The bot to send the conversation to
            conversation_id: Optional conversation ID for history tracking
            save_history: Whether to save the conversation to history
            
        Yields:
            Partial responses from the bot
        """
        await self._ensure_history_initialized()
        
        # Create a new conversation if none provided and history is enabled
        if conversation_id is None and save_history and self.enable_history:
            conversation_id = await self.history.create_conversation(
                title=f"Multi-turn chat with {bot_name}",
                bot_name=bot_name,
                chat_mode="chatbot",
                topic=None  # Topic will be generated from first message if needed
            )
        
        # Reset content processor state for new request
        self.content_processor.last_generating_message = ""
        
        # Save messages to history if needed (with proper role names)
        if save_history and conversation_id and self.enable_history:
            for msg in messages:
                await self.history.add_message(
                    conversation_id=conversation_id,
                    role=self._convert_role_for_history(msg["role"]),
                    content=msg["content"],
                    bot_name=bot_name
                )
        
        # Stream the response with content filtering
        full_response = ""
        try:
            async for chunk in self._stream(bot_name, messages):
                yield chunk
                full_response += chunk
        except Exception as e:
            translated = await self._provider_error(e, bot_name)
            if translated is e:
                raise
            raise translated from e

        # Save the bot response to history
        if save_history and conversation_id and full_response and self.enable_history:
            await self.history.add_message(
                conversation_id=conversation_id,
                role="assistant",  # Save as assistant in history
                content=full_response,
                bot_name=bot_name
            )

    def provider_for_model(self, model: str) -> ProviderSpec:
        """Which provider serves ``model`` (roster-driven; Poe for unknowns)."""
        return get_provider(provider_for(model))

    async def _stream(
        self, model: str, messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """Stream a completion from whichever provider owns ``model``.

        Routing is per model, so a Poe model and an OpenRouter model can be in
        flight side by side (e.g. the two sides of a debate). Poe keeps its
        native ``fastapi_poe`` transport rather than moving to the
        OpenAI-compatible path, because the media bots return markdown media
        links that :class:`ContentProcessor` is built to handle.
        """
        spec = self.provider_for_model(model)
        api_key = api_key_for(spec, self.config)
        if not api_key:
            raise ValueError(
                f"{spec.label} is not configured — set {spec.api_key_env} to use "
                f"'{model}' (key from {spec.signup_url})."
            )

        if spec.name == OPENROUTER:
            max_tokens = getattr(self.config, "openrouter_max_tokens", 0) or None
            async for chunk in stream_openai_compatible(
                spec,
                api_key=api_key,
                model=model,
                # OpenAI-compatible providers use "assistant" natively; the
                # assistant->bot rename is a Poe protocol quirk.
                messages=[
                    {"role": msg["role"], "content": msg["content"]} for msg in messages
                ],
                max_tokens=max_tokens,
            ):
                yield chunk
        else:
            poe_messages = [
                fp.ProtocolMessage(
                    role=self._convert_role_for_api(msg["role"]),
                    content=msg["content"],
                )
                for msg in messages
            ]
            async for partial in fp.get_bot_response(
                messages=poe_messages, bot_name=model, api_key=api_key
            ):
                if hasattr(partial, "text") and partial.text:
                    # Poe streams literal "Generating..." placeholder chunks.
                    if not self.content_processor.should_filter_chunk(partial.text):
                        yield partial.text

        # A completed stream is the cheapest possible evidence that this
        # provider's API access is live; /status reads it instead of probing.
        health.for_provider(spec.name).record_success(source="chat")

    async def _provider_error(self, exc: Exception, model: str) -> Exception:
        """Record a failed call and return the exception to raise.

        Every failure is classified into the stable taxonomy in
        :mod:`pypoe.core.provider_health` and recorded against the provider that
        actually failed, so ``/status`` can report an account-level outage (a
        lapsed subscription, exhausted credits, a revoked key) instead of
        claiming ``ready`` while every chat fails — and so one dead provider
        does not implicate a healthy one. Returns rather than raises so the
        caller's ``raise`` keeps the control flow visible.
        """
        spec = self.provider_for_model(model)
        reason, message = classify_exception(exc)

        # Model-specific failures say nothing about provider health, so they
        # are deliberately left *unrecorded* rather than recorded-then-cleared:
        # writing any observation here would reset the tracker's freshness
        # clock and could bury a genuine outage seen moments earlier.
        if reason in MODEL_SCOPED_REASONS:
            available = [m for m in await self.get_available_bots()
                         if self.provider_for_model(m).name == spec.name]
            if "Cannot access private bots" in str(exc):
                alternatives = [b for b in available if "laude" in b][:3] or available[:5]
                headline = f"Model '{model}' is not accessible (private or deprecated)."
            else:
                alternatives = available[:5]
                headline = f"Model '{model}' does not exist on {spec.label}."
            listing = "".join(f"  • {alt}\n" for alt in alternatives)
            return ValueError(f"{headline}\n\nTry these models instead:\n{listing}")

        health.for_provider(spec.name).record_failure(reason, message, source="chat")

        if reason in ACCOUNT_BLOCKING_REASONS:
            hint = recovery_hint(spec.name, reason)
            return ValueError(f"{message}\n\n{hint}" if hint else message)

        # Rate limits, transport failures, and anything else the provider
        # reported keep their original exception so callers can retry.
        return exc

    async def get_available_bots(self) -> List[str]:
        """Get configured Poe models and the cached, accessible OpenRouter catalog."""
        if getattr(self.config, "openrouter_auto_refresh_models", True):
            await refresh_openrouter_catalog(getattr(self.config, "openrouter_api_key", ""))
        return list(CHAT_MODELS)

    async def get_conversations(self) -> List[Dict[str, Any]]:
        """Get all conversations from history."""
        if not self.enable_history:
            return []
        await self._ensure_history_initialized()
        return await self.history.get_conversations()

    async def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages from a specific conversation."""
        if not self.enable_history:
            return []
        await self._ensure_history_initialized()
        return await self.history.get_conversation_messages(conversation_id)

    async def delete_conversation(self, conversation_id: str):
        """Delete a conversation and all its messages."""
        if not self.enable_history:
            return
        await self._ensure_history_initialized()
        await self.history.delete_conversation(conversation_id)

    async def create_conversation(
        self,
        title: str,
        bot_name: str,
        chat_mode: str = "chatbot",
        topic: str = None,
        bot_names: Optional[List[str]] = None,
        bot_assignments: Optional[Dict[str, Dict[str, Any]]] = None,
        debate_topic: Optional[str] = None,
    ) -> str:
        """
        Create a new conversation with optional topic.

        Args:
            title: The conversation title
            bot_name: The (primary) bot name for this conversation
            chat_mode: The chat mode (chatbot, group, debate)
            topic: Optional topic for the conversation
            bot_names: Required for chat_mode in ('group','debate'); the
                ordered participant list (2-3 entries). Ignored otherwise.
            bot_assignments: Required for chat_mode='debate'; maps each
                participant to its assigned stance.
            debate_topic: Required for chat_mode='debate'; the pinned
                topic prepended to every model's system prompt.

        Returns:
            The conversation ID
        """
        if not self.enable_history:
            raise ValueError("History is not enabled")
        await self._ensure_history_initialized()
        return await self.history.create_conversation(
            title=title,
            bot_name=bot_name,
            chat_mode=chat_mode,
            topic=topic,
            bot_names=bot_names,
            bot_assignments=bot_assignments,
            debate_topic=debate_topic,
        )

    async def generate_topic_from_message(self, first_message: str, bot_name: str = DEFAULT_CHAT_MODEL) -> str:
        """
        Generate a short topic (less than 5 words) from the first user message.
        
        Args:
            first_message: The first message to generate a topic from
            bot_name: The bot to use for topic generation
            
        Returns:
            A short topic string
        """
        try:
            # Try the requested/default model first, then the configured catalog.
            models_to_try = [bot_name] + [model for model in CHAT_MODELS if model != bot_name]

            for model in models_to_try:
                try:
                    # Use a fast model to generate the topic
                    topic_prompt = f"Summarize this question/message in exactly 3-4 words (no more than 5 words): '{first_message}'"
                    
                    full_response = ""
                    async for chunk in self.send_message(
                        message=topic_prompt,
                        bot_name=model,
                        save_history=False  # Don't save this internal conversation
                    ):
                        full_response += chunk
                    
                    # Clean up the response - remove quotes, extra punctuation
                    topic = full_response.strip().strip('"').strip("'").strip('.').strip()
                    
                    # Ensure it's not too long (max 5 words)
                    words = topic.split()
                    if len(words) > 5:
                        topic = ' '.join(words[:5])
                    
                    if topic and topic.lower() not in ['error', 'failed', 'sorry', 'cannot']:
                        print(f"Generated topic using {model}: '{topic}'")
                        return topic
                        
                except Exception as model_error:
                    print(f"Failed to generate topic with {model}: {model_error}")
                    continue
            
            # If all models fail, use fallback
            print("All models failed for topic generation, using fallback")
            return self._generate_fallback_topic(first_message)
            
        except Exception as e:
            print(f"Warning: Failed to generate topic: {e}")
            return self._generate_fallback_topic(first_message)

    def _generate_fallback_topic(self, first_message: str) -> str:
        """
        Generate a simple fallback topic from the first message.
        
        Args:
            first_message: The message to generate a topic from
            
        Returns:
            A short topic string
        """
        try:
            # Remove common words and punctuation
            import re
            cleaned_message = re.sub(r'[^\w\s]', '', first_message.lower())
            
            # Split into words and filter out common words
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must', 'shall', 'what', 'when', 'where', 'why', 'how', 'who', 'which', 'that', 'this', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'hello', 'hi', 'hey', 'please', 'help', 'thanks', 'thank'}
            
            words = [word for word in cleaned_message.split() if word not in common_words and len(word) > 2]
            
            # Take first 3-4 meaningful words
            if words:
                topic = ' '.join(words[:4])
                return topic.title()  # Capitalize first letter of each word
            
            # If no meaningful words, use first few characters
            if len(first_message) > 10:
                return first_message[:15].strip().title()
            else:
                return first_message.strip().title()
                
        except Exception as e:
            print(f"Warning: Fallback topic generation failed: {e}")
            # Last resort: use first few words
            words = first_message.split()[:3]
            return ' '.join(words) if words else "Chat Topic"

    async def update_conversation_topic(self, conversation_id: str, topic: str):
        """
        Update the topic of an existing conversation.
        
        Args:
            conversation_id: The conversation ID to update
            topic: The new topic for the conversation
        """
        if not self.enable_history:
            raise ValueError("History is not enabled")
        await self._ensure_history_initialized()
        
        import aiosqlite
        async with self.history._lock:
            async with aiosqlite.connect(self.history.db_path) as db:
                # Owner-scoping (§4.9): scope the update to this client's owner
                # when one is bound; unscoped clients update by id only.
                if self._owner is not None:
                    await db.execute(
                        "UPDATE conversations SET topic = ? "
                        "WHERE id = ? AND owner = ?",
                        (topic, conversation_id, self._owner),
                    )
                else:
                    await db.execute(
                        "UPDATE conversations SET topic = ? WHERE id = ?",
                        (topic, conversation_id)
                    )
                await db.commit()

    async def generate_and_update_topic(
        self,
        conversation_id: str,
        first_message: str,
        bot_name: str = DEFAULT_CHAT_MODEL,
    ):
        """
        Generate a topic from the first message and update the conversation.
        
        Args:
            conversation_id: The conversation ID to update
            first_message: The first message to generate a topic from
            bot_name: The bot to use for topic generation
        """
        try:
            topic = await self.generate_topic_from_message(first_message, bot_name)
            await self.update_conversation_topic(conversation_id, topic)
            return topic
        except Exception as e:
            print(f"Failed to generate and update topic: {e}")
            return None

    async def close(self):
        """Clean up resources."""
        if self.enable_history and self._history_initialized:
            await self.history.close() 