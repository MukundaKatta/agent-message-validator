"""Tests for agent-message-validator.

Uses only the Python standard library (``unittest``) so the suite has the
same zero-dependency footprint as the package itself. Run with::

    python3 -m unittest discover -s tests
"""

import os
import sys
import unittest

# Make the src-layout package importable whether or not it has been installed.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from agent_message_validator import (  # noqa: E402
    MessageValidationError,
    ValidationResult,
    check_messages,
    is_valid,
    validate_messages,
)


class EmptyListTests(unittest.TestCase):
    def test_empty_messages_ok(self):
        result = validate_messages([])
        self.assertTrue(result.ok)
        self.assertEqual(result.message_count, 0)

    def test_empty_messages_strict_does_not_raise(self):
        # An empty list is valid; strict mode must not raise.
        result = validate_messages([], strict=True)
        self.assertTrue(result.ok)


class BasicValidStructureTests(unittest.TestCase):
    def test_single_user_message(self):
        self.assertTrue(validate_messages([{"role": "user", "content": "Hello"}]).ok)

    def test_user_assistant_exchange(self):
        msgs = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."},
        ]
        self.assertTrue(validate_messages(msgs).ok)

    def test_multiple_turns(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "Great!"},
        ]
        self.assertTrue(validate_messages(msgs).ok)

    def test_content_as_list(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        self.assertTrue(validate_messages(msgs).ok)


class ConsecutiveRoleTests(unittest.TestCase):
    def test_consecutive_user_messages(self):
        msgs = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
        ]
        result = validate_messages(msgs)
        self.assertFalse(result.ok)
        self.assertTrue(any("consecutive" in e for e in result.errors))

    def test_consecutive_assistant_messages(self):
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A1"},
            {"role": "assistant", "content": "A2"},
        ]
        self.assertFalse(validate_messages(msgs).ok)


class FirstMessageRoleTests(unittest.TestCase):
    def test_first_message_assistant_error(self):
        msgs = [
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Hi"},
        ]
        result = validate_messages(msgs)
        self.assertFalse(result.ok)
        self.assertTrue(any("first message" in e for e in result.errors))

    def test_first_message_user_ok(self):
        self.assertTrue(validate_messages([{"role": "user", "content": "Hello"}]).ok)


class MissingKeyTests(unittest.TestCase):
    def test_missing_role(self):
        result = validate_messages([{"content": "No role here"}])
        self.assertFalse(result.ok)
        self.assertTrue(any("'role'" in e for e in result.errors))

    def test_missing_content(self):
        result = validate_messages([{"role": "user"}])
        self.assertFalse(result.ok)
        self.assertTrue(any("'content'" in e for e in result.errors))


class EmptyContentTests(unittest.TestCase):
    def test_empty_string_content(self):
        result = validate_messages([{"role": "user", "content": ""}])
        self.assertFalse(result.ok)
        self.assertTrue(any("empty content" in e for e in result.errors))

    def test_whitespace_only_content(self):
        self.assertFalse(validate_messages([{"role": "user", "content": "   "}]).ok)

    def test_empty_list_content(self):
        self.assertFalse(validate_messages([{"role": "user", "content": []}]).ok)

    def test_allow_empty_content(self):
        result = validate_messages([{"role": "user", "content": ""}], allow_empty_content=True)
        self.assertTrue(result.ok)


class ToolPairingTests(unittest.TestCase):
    def test_valid_tool_use_result_pair(self):
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
        self.assertTrue(validate_messages(msgs).ok)

    def test_unmatched_tool_use_error(self):
        msgs = [
            {"role": "user", "content": "Go"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_001", "name": "search", "input": {}},
                ],
            },
            {"role": "user", "content": "Okay"},
        ]
        result = validate_messages(msgs)
        self.assertFalse(result.ok)
        self.assertTrue(any("tu_001" in e for e in result.errors))

    def test_orphan_tool_result_error(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_999", "content": "value"},
            ]},
        ]
        result = validate_messages(msgs)
        self.assertFalse(result.ok)
        self.assertTrue(any("tu_999" in e for e in result.errors))

    def test_multiple_tool_uses_all_resolved(self):
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
        self.assertTrue(validate_messages(msgs).ok)

    def test_multiple_tool_uses_one_unresolved(self):
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
                ],
            },
            {"role": "assistant", "content": "Done."},
        ]
        result = validate_messages(msgs)
        self.assertFalse(result.ok)
        self.assertTrue(any("t2" in e for e in result.errors))

    def test_text_and_tool_use_in_assistant(self):
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
        self.assertTrue(validate_messages(msgs).ok)


class MultipleErrorTests(unittest.TestCase):
    def test_multiple_errors_all_reported(self):
        msgs = [
            {"role": "assistant", "content": "Hello"},  # first message is assistant
            {"role": "assistant", "content": "Again"},  # consecutive same role
        ]
        result = validate_messages(msgs)
        self.assertFalse(result.ok)
        self.assertGreaterEqual(result.error_count, 2)


class ValidationResultTests(unittest.TestCase):
    def test_result_bool_true(self):
        self.assertIs(bool(validate_messages([{"role": "user", "content": "Hi"}])), True)

    def test_result_bool_false(self):
        self.assertIs(bool(validate_messages([{"role": "assistant", "content": "Hi"}])), False)

    def test_result_error_count(self):
        result = validate_messages([{"role": "assistant", "content": "Hi"}])
        self.assertGreaterEqual(result.error_count, 1)

    def test_result_message_count(self):
        msgs = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
        ]
        self.assertEqual(validate_messages(msgs).message_count, 2)

    def test_summary_ok(self):
        result = validate_messages([{"role": "user", "content": "Hi"}])
        self.assertIn("passed", result.summary())

    def test_summary_errors(self):
        result = validate_messages([{"role": "assistant", "content": "Hi"}])
        self.assertIn("error", result.summary().lower())

    def test_result_is_dataclass_instance(self):
        result = validate_messages([{"role": "user", "content": "Hi"}])
        self.assertIsInstance(result, ValidationResult)


class StrictModeTests(unittest.TestCase):
    def test_strict_raises(self):
        with self.assertRaises(MessageValidationError):
            validate_messages([{"role": "assistant", "content": "Hello"}], strict=True)

    def test_strict_no_raise_on_valid(self):
        result = validate_messages([{"role": "user", "content": "Hello"}], strict=True)
        self.assertTrue(result.ok)

    def test_check_messages_raises(self):
        with self.assertRaises(MessageValidationError):
            check_messages([{"role": "assistant", "content": "Bad"}])

    def test_check_messages_passes(self):
        check_messages([{"role": "user", "content": "Good"}])  # no exception

    def test_check_messages_returns_none(self):
        self.assertIsNone(check_messages([{"role": "user", "content": "Good"}]))


class IsValidTests(unittest.TestCase):
    def test_is_valid_true(self):
        self.assertIs(is_valid([{"role": "user", "content": "Hi"}]), True)

    def test_is_valid_false(self):
        self.assertIs(is_valid([{"role": "assistant", "content": "Hi"}]), False)

    def test_is_valid_passes_kwargs(self):
        # allow_empty_content should flow through is_valid().
        self.assertTrue(is_valid([{"role": "user", "content": ""}], allow_empty_content=True))


class MalformedInputTests(unittest.TestCase):
    """The validator must report malformed input cleanly, never crash on it."""

    def test_non_list_top_level(self):
        result = validate_messages("not a list")
        self.assertFalse(result.ok)
        self.assertTrue(any("must be a list" in e for e in result.errors))

    def test_single_dict_instead_of_list(self):
        # A common mistake: passing one message dict directly.
        result = validate_messages({"role": "user", "content": "hi"})
        self.assertFalse(result.ok)
        self.assertTrue(any("must be a list" in e for e in result.errors))

    def test_non_list_strict_raises(self):
        with self.assertRaises(MessageValidationError):
            validate_messages(None, strict=True)

    def test_tuple_is_accepted(self):
        result = validate_messages(({"role": "user", "content": "hi"},))
        self.assertTrue(result.ok)

    def test_non_dict_message_item(self):
        result = validate_messages([{"role": "user", "content": "hi"}, "oops"])
        self.assertFalse(result.ok)
        self.assertTrue(any("must be a dict" in e for e in result.errors))

    def test_non_dict_message_does_not_crash_pairing(self):
        # A non-dict entry sitting between valid tool_use/tool_result must not
        # raise AttributeError during pairing.
        msgs = [
            {"role": "user", "content": "Go"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
            ]},
            None,
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "r"},
            ]},
        ]
        result = validate_messages(msgs)
        # The None item is flagged, but the tool pair still resolves cleanly.
        self.assertTrue(any("must be a dict" in e for e in result.errors))
        self.assertFalse(any("t1" in e for e in result.errors))

    def test_non_dict_content_block_does_not_crash(self):
        # A bare string inside a content list must not raise.
        result = validate_messages([{"role": "user", "content": ["just a string block"]}])
        self.assertTrue(result.ok)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
