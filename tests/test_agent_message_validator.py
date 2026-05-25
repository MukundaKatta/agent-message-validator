"""Tests for agent-message-validator."""

import pytest

from agent_message_validator import (
    MessageValidationError,
    ValidationResult,
    check_messages,
    is_valid,
    validate_messages,
)


# ---------------------------------------------------------------------------
# Empty list
# ---------------------------------------------------------------------------

def test_empty_messages_ok():
    result = validate_messages([])
    assert result.ok
    assert result.message_count == 0

# ---------------------------------------------------------------------------
# Basic valid structures
# ---------------------------------------------------------------------------

def test_single_user_message():
    msgs = [{"role": "user", "content": "Hello"}]
    assert validate_messages(msgs).ok

def test_user_assistant_exchange():
    msgs = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ]
    assert validate_messages(msgs).ok

def test_multiple_turns():
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "Great!"},
    ]
    assert validate_messages(msgs).ok

def test_content_as_list():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
    ]
    assert validate_messages(msgs).ok

# ---------------------------------------------------------------------------
# Consecutive same-role error
# ---------------------------------------------------------------------------

def test_consecutive_user_messages():
    msgs = [
        {"role": "user", "content": "First"},
        {"role": "user", "content": "Second"},
    ]
    result = validate_messages(msgs)
    assert not result.ok
    assert any("consecutive" in e for e in result.errors)

def test_consecutive_assistant_messages():
    msgs = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A1"},
        {"role": "assistant", "content": "A2"},
    ]
    result = validate_messages(msgs)
    assert not result.ok

# ---------------------------------------------------------------------------
# First message must be user
# ---------------------------------------------------------------------------

def test_first_message_assistant_error():
    msgs = [
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Hi"},
    ]
    result = validate_messages(msgs)
    assert not result.ok
    assert any("first message" in e for e in result.errors)

def test_first_message_user_ok():
    msgs = [{"role": "user", "content": "Hello"}]
    assert validate_messages(msgs).ok

# ---------------------------------------------------------------------------
# Missing keys
# ---------------------------------------------------------------------------

def test_missing_role():
    msgs = [{"content": "No role here"}]
    result = validate_messages(msgs)
    assert not result.ok
    assert any("'role'" in e for e in result.errors)

def test_missing_content():
    msgs = [{"role": "user"}]
    result = validate_messages(msgs)
    assert not result.ok
    assert any("'content'" in e for e in result.errors)

# ---------------------------------------------------------------------------
# Empty content
# ---------------------------------------------------------------------------

def test_empty_string_content():
    msgs = [{"role": "user", "content": ""}]
    result = validate_messages(msgs)
    assert not result.ok
    assert any("empty content" in e for e in result.errors)

def test_whitespace_only_content():
    msgs = [{"role": "user", "content": "   "}]
    result = validate_messages(msgs)
    assert not result.ok

def test_empty_list_content():
    msgs = [{"role": "user", "content": []}]
    result = validate_messages(msgs)
    assert not result.ok

def test_allow_empty_content():
    msgs = [{"role": "user", "content": ""}]
    result = validate_messages(msgs, allow_empty_content=True)
    assert result.ok

# ---------------------------------------------------------------------------
# tool_use / tool_result pairing
# ---------------------------------------------------------------------------

def test_valid_tool_use_result_pair():
    msgs = [
        {"role": "user", "content": "Search for Paris"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_001", "name": "search", "input": {"q": "Paris"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_001", "content": "Paris is the capital of France"},
            ],
        },
        {"role": "assistant", "content": "Paris is the capital of France."},
    ]
    assert validate_messages(msgs).ok

def test_unmatched_tool_use_error():
    msgs = [
        {"role": "user", "content": "Go"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_001", "name": "search", "input": {}},
            ],
        },
        # No tool_result follows
        {"role": "user", "content": "Okay"},
    ]
    result = validate_messages(msgs)
    assert not result.ok
    assert any("tu_001" in e for e in result.errors)

def test_orphan_tool_result_error():
    msgs = [
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_999", "content": "value"},
        ]},
    ]
    result = validate_messages(msgs)
    assert not result.ok
    assert any("tu_999" in e for e in result.errors)

def test_multiple_tool_uses_all_resolved():
    msgs = [
        {"role": "user", "content": "Go"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
                {"type": "tool_use", "id": "t2", "name": "b", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "r1"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "r2"},
            ],
        },
        {"role": "assistant", "content": "Done."},
    ]
    assert validate_messages(msgs).ok

def test_multiple_tool_uses_one_unresolved():
    msgs = [
        {"role": "user", "content": "Go"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
                {"type": "tool_use", "id": "t2", "name": "b", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "r1"},
                # t2 missing
            ],
        },
        {"role": "assistant", "content": "Done."},
    ]
    result = validate_messages(msgs)
    assert not result.ok
    assert any("t2" in e for e in result.errors)

# ---------------------------------------------------------------------------
# Multiple errors reported together
# ---------------------------------------------------------------------------

def test_multiple_errors_all_reported():
    msgs = [
        {"role": "assistant", "content": "Hello"},  # first message is assistant
        {"role": "assistant", "content": "Again"},  # consecutive same role
    ]
    result = validate_messages(msgs)
    assert not result.ok
    assert result.error_count >= 2

# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

def test_result_bool_true():
    result = validate_messages([{"role": "user", "content": "Hi"}])
    assert bool(result) is True

def test_result_bool_false():
    result = validate_messages([{"role": "assistant", "content": "Hi"}])
    assert bool(result) is False

def test_result_error_count():
    msgs = [{"role": "assistant", "content": "Hi"}]
    result = validate_messages(msgs)
    assert result.error_count >= 1

def test_result_message_count():
    msgs = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
    ]
    result = validate_messages(msgs)
    assert result.message_count == 2

def test_summary_ok():
    result = validate_messages([{"role": "user", "content": "Hi"}])
    assert "passed" in result.summary()

def test_summary_errors():
    result = validate_messages([{"role": "assistant", "content": "Hi"}])
    assert "error" in result.summary().lower()

# ---------------------------------------------------------------------------
# strict mode
# ---------------------------------------------------------------------------

def test_strict_raises():
    msgs = [{"role": "assistant", "content": "Hello"}]
    with pytest.raises(MessageValidationError):
        validate_messages(msgs, strict=True)

def test_strict_no_raise_on_valid():
    msgs = [{"role": "user", "content": "Hello"}]
    result = validate_messages(msgs, strict=True)
    assert result.ok

def test_check_messages_raises():
    msgs = [{"role": "assistant", "content": "Bad"}]
    with pytest.raises(MessageValidationError):
        check_messages(msgs)

def test_check_messages_passes():
    msgs = [{"role": "user", "content": "Good"}]
    check_messages(msgs)  # no exception

# ---------------------------------------------------------------------------
# is_valid shorthand
# ---------------------------------------------------------------------------

def test_is_valid_true():
    assert is_valid([{"role": "user", "content": "Hi"}]) is True

def test_is_valid_false():
    assert is_valid([{"role": "assistant", "content": "Hi"}]) is False

# ---------------------------------------------------------------------------
# text + tool_use mixed content
# ---------------------------------------------------------------------------

def test_text_and_tool_use_in_assistant():
    msgs = [
        {"role": "user", "content": "Search and summarize"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me search."},
                {"type": "tool_use", "id": "tu_x", "name": "search", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_x", "content": "results"},
            ],
        },
        {"role": "assistant", "content": "Here's the summary."},
    ]
    assert validate_messages(msgs).ok
