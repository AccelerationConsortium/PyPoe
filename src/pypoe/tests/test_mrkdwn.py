"""Unit tests for ``pypoe.core.mrkdwn``.

The converter sits at the Slack posting boundary, where it sees a mix of
CommonMark (model output, plus PyPoe's own ``**bold**`` literals) and
hand-written mrkdwn. The three properties the call sites depend on —
correct conversion, idempotency, and code-span safety — are covered here.
"""

from __future__ import annotations

import pytest

from pypoe.core.mrkdwn import to_mrkdwn


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected",
    [
        # Bold: the reported symptom — `**text**` shown literally in Slack.
        ("**Root cause:** Wi-Fi drop", "*Root cause:* Wi-Fi drop"),
        ("a **b** c **d** e", "a *b* c *d* e"),
        # Strikethrough halves its tildes too.
        ("~~stale~~ current", "~stale~ current"),
        # Every heading level collapses to bold; mrkdwn has no headings.
        ("# Summary", "*Summary*"),
        ("###### Deep", "*Deep*"),
        ("## Closed ##", "*Closed*"),
        # Bullets become real bullet glyphs, indentation preserved.
        ("- one\n- two", "•  one\n•  two"),
        ("* star", "•  star"),
        ("+ plus", "•  plus"),
        ("    - nested", "    •  nested"),
        # Links take Slack's <url|label> form.
        ("see [docs](http://x/y)", "see <http://x/y|docs>"),
        ("![shot](http://x/i.png)", "<http://x/i.png|shot>"),
        ("[](http://x/y)", "<http://x/y>"),
        # Combined, as a model actually writes an incident report.
        ("- **Device**: `ot2_hte`", "•  *Device*: `ot2_hte`"),
    ],
)
def test_converts(source: str, expected: str) -> None:
    assert to_mrkdwn(source) == expected


# --------------------------------------------------------------------------
# What must survive untouched
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # Already-correct mrkdwn from PyPoe's own alert lines. Rewriting
        # single asterisks would corrupt these to fix a rarer case.
        ":white_check_mark: *SDL Assistant* recovered.",
        ":rotating_light: *ot2_hte* DOWN — investigating…",
        # Slack's own syntax: mentions, channel refs, existing links.
        "<@U123> please check <#C456> and <http://x/y|the runbook>",
        # Dunders stay dunders — the reason `__bold__` is not converted.
        "traceback in __init__.py",
        # Ordered lists and quotes render acceptably as-is.
        "1. first\n2. second",
        "> quoted line",
        # Bare asterisk with no closing partner.
        "2 ** 8 is 256",
    ],
)
def test_passthrough(text: str) -> None:
    assert to_mrkdwn(text) == text


def test_bullet_rule_does_not_eat_a_leading_bold_span() -> None:
    """``*SDL Dashboard* recovered`` opens with ``*`` but is not a bullet.

    The bullet pattern requires whitespace after the marker, which is the
    only thing separating these two cases.
    """
    assert to_mrkdwn("*SDL Dashboard* recovered.") == "*SDL Dashboard* recovered."


# --------------------------------------------------------------------------
# Code spans
# --------------------------------------------------------------------------


def test_fenced_block_is_verbatim() -> None:
    src = "before\n```\ndef f(**kwargs):\n    # **not bold**\n```\nafter **bold**"
    out = to_mrkdwn(src)
    assert "def f(**kwargs):" in out
    assert "# **not bold**" in out
    assert out.endswith("after *bold*")


def test_inline_code_is_verbatim() -> None:
    assert to_mrkdwn("pass `**kwargs` through") == "pass `**kwargs` through"


def test_prose_around_code_still_converts() -> None:
    assert to_mrkdwn("**a** `**b**` **c**") == "*a* `**b**` *c*"


# --------------------------------------------------------------------------
# Idempotency — _post_slack carries converted and unconverted text alike
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "**bold** and [link](http://x) and\n- bullet\n# head",
        ":white_check_mark: *SDL Assistant* recovered.",
        "```\n**code**\n```\n**prose**",
    ],
)
def test_idempotent(text: str) -> None:
    once = to_mrkdwn(text)
    assert to_mrkdwn(once) == once


# --------------------------------------------------------------------------
# Degenerate input — call sites pass payload fields through unchecked
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["", None, 42, {"text": "x"}])
def test_non_text_input_is_returned_unchanged(value) -> None:
    assert to_mrkdwn(value) is value


def test_bold_does_not_span_lines() -> None:
    """An unterminated ``**`` must not swallow the rest of the message."""
    src = "**unclosed\nnext line **also unclosed"
    assert to_mrkdwn(src) == src
