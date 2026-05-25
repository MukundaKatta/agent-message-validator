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
    """Return content as a list of blocks regardless of string/list form."""
    content = message.get("content", "")
    if isinstance(content, str):
        if content.strip():
            return [{"type": "text", "text": content}]
        return []
    if isinstance(content, list):
        return content
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
    1. No consecutive messages with the same role.
    2. Every tool_use block has a matching tool_result in the next message.
    3. No orphan tool_result blocks (no prior tool_use).
    4. Content is not empty (unless allow_empty_content=True).
    5. First message role is 'user' (not 'assistant').
    6. All messages have a 'role' key.
    7. All messages have a 'content' key.

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

    if not messages:
        result = ValidationResult(ok=True, errors=[], message_count=0)
        if strict and not result.ok:
            raise MessageValidationError(result.summary())
        return result

    # 1. All messages have 'role'
    for i, msg in enumerate(messages):
        if "role" not in msg:
            errors.append(f"message[{i}] missing 'role' key")

    # 2. All messages have 'content'
    for i, msg in enumerate(messages):
        if "content" not in msg:
            errors.append(f"message[{i}] missing 'content' key")

    # 3. First message must be 'user'
    first_role = messages[0].get("role")
    if first_role == "assistant":
        errors.append("first message must have role 'user', not 'assistant'")

    # 4. No consecutive same-role messages
    for i in range(1, len(messages)):
        prev_role = messages[i - 1].get("role")
        curr_role = messages[i].get("role")
        if prev_role and curr_role and prev_role == curr_role:
            errors.append(
                f"consecutive same-role messages at index {i - 1} and {i} (role='{curr_role}')"
            )

    # 5. Empty content check
    if not allow_empty_content:
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if isinstance(content, str) and not content.strip():
                errors.append(f"message[{i}] has empty content (role='{msg.get('role', '?')}')")
            elif isinstance(content, list) and len(content) == 0:
                errors.append(f"message[{i}] has empty content list (role='{msg.get('role', '?')}')")

    # 6. tool_use / tool_result pairing
    # Collect all tool_use ids emitted by assistant messages
    pending_tool_use_ids: list[str] = []  # ids waiting for a result

    for i, msg in enumerate(messages):
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
