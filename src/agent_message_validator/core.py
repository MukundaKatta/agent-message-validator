"""Validate Anthropic-style message lists before sending to the API.

Catches structural errors that would cause 400-level API rejections:
 - consecutive same-role messages
 - tool_use blocks without a matching tool_result
 - orphan tool_result blocks
 - empty content
 - first message from assistant

Zero dependencies — standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class MessageValidationError(Exception):
    """Raised when validate() is called with strict=True and errors are found."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Return content as a list of dict blocks regardless of string/list form.

    Non-dict blocks inside a content list are ignored here so that callers can
    iterate safely; structural validation of block shape is handled separately.
    """
    content = message.get("content", "")
    if isinstance(content, str):
        if content.strip():
            return [{"type": "text", "text": content}]
        return []
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _tool_use_ids(message: dict[str, Any]) -> list[str]:
    return [
        b.get("id", "")
        for b in _content_blocks(message)
        if b.get("type") == "tool_use"
    ]


def _tool_result_ids(message: dict[str, Any]) -> list[str]:
    return [
        b.get("tool_use_id", "")
        for b in _content_blocks(message)
        if b.get("type") == "tool_result"
    ]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of validating a message list.

    Attributes:
        ok: True if all checks passed.
        errors: list of human-readable error strings.
        message_count: number of messages validated.
    """

    ok: bool
    errors: list[str]
    message_count: int

    @property
    def error_count(self) -> int:
        return len(self.errors)

    def summary(self) -> str:
        if self.ok:
            return f"All checks passed ({self.message_count} messages)."
        lines = [f"{self.error_count} error(s) in {self.message_count} message(s):"]
        for e in self.errors:
            lines.append(f"  - {e}")
        return "\n".join(lines)

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def validate_messages(
    messages: list[dict[str, Any]],
    *,
    strict: bool = False,
    allow_empty_content: bool = False,
) -> ValidationResult:
    """Validate an Anthropic-style message list.

    Checks performed:
    0. ``messages`` is a list and every item is a dict.
    1. No consecutive messages with the same role.
    2. Every tool_use block has a matching tool_result.
    3. No orphan tool_result blocks (no prior tool_use).
    4. Content is not empty (unless allow_empty_content=True).
    5. First message role is 'user' (not 'assistant').
    6. All messages have a 'role' key.
    7. All messages have a 'content' key.

    The function never raises on malformed input unless ``strict`` is True: a
    non-list argument or a non-dict message produces an ordinary error in the
    result rather than a TypeError/AttributeError.

    Args:
        messages: list of message dicts.
        strict: if True, raise MessageValidationError on any error.
        allow_empty_content: if True, skip empty-content checks.

    Returns:
        ValidationResult with ok, errors, message_count.

    Raises:
        MessageValidationError: if strict=True and errors are found.
    """
    errors: list[str] = []

    # 0. Top-level container must be a list (tuples are accepted as list-like).
    if not isinstance(messages, (list, tuple)):
        result = ValidationResult(
            ok=False,
            errors=[f"messages must be a list, got {type(messages).__name__}"],
            message_count=0,
        )
        if strict:
            raise MessageValidationError(result.summary())
        return result

    if not messages:
        result = ValidationResult(ok=True, errors=[], message_count=0)
        # An empty list is valid, so strict mode never raises here.
        return result

    # 0b. Every item must be a dict; non-dict items are reported and skipped by
    # the structural checks below so a single bad entry cannot crash validation.
    non_dict_indices = {i for i, msg in enumerate(messages) if not isinstance(msg, dict)}
    for i in sorted(non_dict_indices):
        errors.append(
            f"message[{i}] must be a dict, got {type(messages[i]).__name__}"
        )

    def _role_at(index: int) -> Any:
        """Role of a message, or None for non-dict messages."""
        msg = messages[index]
        return msg.get("role") if isinstance(msg, dict) else None

    # 1. All messages have 'role'
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and "role" not in msg:
            errors.append(f"message[{i}] missing 'role' key")

    # 2. All messages have 'content'
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and "content" not in msg:
            errors.append(f"message[{i}] missing 'content' key")

    # 3. First message must be 'user'
    if _role_at(0) == "assistant":
        errors.append("first message must have role 'user', not 'assistant'")

    # 4. No consecutive same-role messages
    for i in range(1, len(messages)):
        prev_role = _role_at(i - 1)
        curr_role = _role_at(i)
        if prev_role and curr_role and prev_role == curr_role:
            errors.append(
                f"consecutive same-role messages at index {i - 1} and {i} (role='{curr_role}')"
            )

    # 5. Empty content check
    if not allow_empty_content:
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and not content.strip():
                errors.append(f"message[{i}] has empty content (role='{msg.get('role', '?')}')")
            elif isinstance(content, list) and len(content) == 0:
                errors.append(f"message[{i}] has empty content list (role='{msg.get('role', '?')}')")

    # 6. tool_use / tool_result pairing
    # Collect all tool_use ids emitted by assistant messages
    pending_tool_use_ids: list[str] = []  # ids waiting for a result

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")

        if role == "assistant":
            use_ids = _tool_use_ids(msg)
            pending_tool_use_ids.extend(use_ids)

        elif role == "user":
            result_ids = _tool_result_ids(msg)
            for rid in result_ids:
                if rid in pending_tool_use_ids:
                    pending_tool_use_ids.remove(rid)
                else:
                    errors.append(
                        f"message[{i}] tool_result for '{rid}' has no matching tool_use"
                    )

    for unmatched_id in pending_tool_use_ids:
        errors.append(f"tool_use '{unmatched_id}' has no matching tool_result")

    result = ValidationResult(ok=len(errors) == 0, errors=errors, message_count=len(messages))

    if strict and not result.ok:
        raise MessageValidationError(result.summary())

    return result


def is_valid(messages: list[dict[str, Any]], **kwargs: Any) -> bool:
    """Return True if the message list passes validation."""
    return validate_messages(messages, **kwargs).ok


def check_messages(messages: list[dict[str, Any]], **kwargs: Any) -> None:
    """Validate and raise MessageValidationError if invalid."""
    validate_messages(messages, strict=True, **kwargs)
