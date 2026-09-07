"""Markdown → Slack *mrkdwn* conversion.

Slack does not speak CommonMark. Its ``text`` fields are rendered as
**mrkdwn**, a different dialect whose most visible divergence is bold:
CommonMark writes ``**bold**``, mrkdwn writes ``*bold*``. Posting
CommonMark verbatim therefore shows the literal asterisks, which is what
every PyPoe Slack surface did before this module existed.

Two independent sources produce CommonMark on the way to Slack:

* **PyPoe's own strings** — the Slack bot's help, stats, and response
  headers are written with ``**...**`` (48 occurrences in ``bot.py``).
* **Model output** — chat replies and the ``claude -p`` incident reports
  in :mod:`pypoe.lab.alert_routes` are markdown by default; no prompt can
  reliably suppress that.

Rather than hand-editing every literal and hoping models comply, callers
run :func:`to_mrkdwn` at the *posting boundary* (the ``chat_postMessage`` /
``chat_update`` call sites). One chokepoint, both sources fixed.

Conversions applied, outside code spans:

===========================  ==========================
CommonMark                   mrkdwn
===========================  ==========================
``**bold**``                 ``*bold*``
``~~strike~~``               ``~strike~``
``# Heading``                ``*Heading*``
``- item`` / ``* item``      ``•  item``
``[text](url)``              ``<url|text>``
===========================  ==========================

Three properties the call sites rely on:

* **Idempotent.** Text already in mrkdwn passes through unchanged — single
  asterisks are never touched, so ``*SDL Assistant* recovered`` survives.
  This matters because ``_post_slack`` carries both hand-written mrkdwn
  and model markdown through the same path.
* **Code-safe.** Fenced blocks and inline spans are held out verbatim, so
  a ``**kwargs`` in a traceback or a markdown sample in a code fence is
  not rewritten.
* **Conservative.** Anything ambiguous is left alone rather than guessed
  at — see the notes below.

Deliberately *not* converted:

* ``*italic*`` → ``_italic_``. A single-asterisk span is ambiguous: it is
  CommonMark italic but mrkdwn *bold*, and PyPoe's own strings already use
  it as bold. Rewriting would corrupt correct text to fix a rarer case.
* ``__bold__``. Indistinguishable from a Python dunder without real
  parsing (``__init__.py`` would become ``*init*.py``), and models emit
  ``**`` overwhelmingly more often.
* Ordered lists and ``>`` quotes. Slack renders both acceptably as-is.
"""

from __future__ import annotations

import re

__all__ = ["to_mrkdwn"]

#: Spans held out of conversion: fenced blocks first (so a ``**`` inside a
#: fence is never rewritten), then double- and single-backtick inline code.
#: Kept as a capturing group so ``re.split`` interleaves code with prose.
_CODE_SPAN = re.compile(r"(```.*?```|``[^`]+``|`[^`\n]+`)", re.DOTALL)

#: ``[text](url)`` → ``<url|text>``. Run before bold so a bolded link label
#: converts cleanly. ``!`` is consumed so an image renders as its alt-text
#: link rather than a stray bang.
_LINK = re.compile(r"!?\[([^\]\n]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

#: ``**bold**`` → ``*bold*``. Anchored on non-space so ``** ``/`` **`` (a
#: literal pair, e.g. in prose about markdown) is left alone, and confined
#: to one line so an unterminated ``**`` cannot swallow a paragraph.
_BOLD = re.compile(r"\*\*(?=\S)([^\n]+?)(?<=\S)\*\*")

#: ``~~strike~~`` → ``~strike~``, same anchoring rationale as bold.
_STRIKE = re.compile(r"~~(?=\S)([^\n]+?)(?<=\S)~~")

#: ATX headings → bold. Slack has no heading level, so all six collapse.
#: Trailing closing hashes (``## Title ##``) are stripped.
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)

#: Bullet markers → ``•``. Requires whitespace after the marker, which is
#: what keeps a line opening with ``*bold*`` from being read as a bullet.
#: Indentation is preserved so nested lists keep their shape.
_BULLET = re.compile(r"^([ \t]*)[-*+][ \t]+", re.MULTILINE)


def _convert_prose(text: str) -> str:
    """Apply every conversion to one non-code segment.

    Order is load-bearing: links before bold (a bold label must survive the
    link rewrite), headings before bullets (so ``# - x`` is a heading, not a
    bullet).
    """
    text = _LINK.sub(lambda m: f"<{m.group(2)}|{m.group(1)}>" if m.group(1) else f"<{m.group(2)}>", text)
    text = _BOLD.sub(r"*\1*", text)
    text = _STRIKE.sub(r"~\1~", text)
    text = _HEADING.sub(r"*\1*", text)
    text = _BULLET.sub(r"\1•  ", text)
    return text


def to_mrkdwn(text: str) -> str:
    """Return ``text`` with CommonMark constructs rewritten as Slack mrkdwn.

    Safe to apply to text that is already mrkdwn, or to a mix of the two —
    see the module docstring for the exact conversion table and the
    constructs deliberately left alone.

    Non-``str`` input (or ``None``) is returned unchanged so a call site can
    pass a Slack payload field through without pre-checking it.
    """
    if not isinstance(text, str) or not text:
        return text
    # re.split with a capturing pattern yields [prose, code, prose, code, ...];
    # odd indices are the held-out code spans and pass through untouched.
    parts = _CODE_SPAN.split(text)
    return "".join(
        part if i % 2 else _convert_prose(part) for i, part in enumerate(parts)
    )
