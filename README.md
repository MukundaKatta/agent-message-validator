# agent-message-validator

Validate Anthropic-style message lists **before** you send them to the API.
Zero dependencies, standard library only.

Multi-turn and tool-using agent loops accumulate a `messages` list over many
iterations. A single structural slip — two assistant turns in a row, a
`tool_use` that never got a `tool_result`, an empty content block — gets
rejected by the API with a 400 error, often deep inside a long-running loop.
This package catches those mistakes locally so you can fail fast (or repair the
list) instead of paying for a round-trip just to learn the request was malformed.

## What it catches

- **Consecutive same-role messages** (`user` followed by `user`, etc.)
- **First message from `assistant`** (the API expects the conversation to open with `user`)
- **`tool_use` without a matching `tool_result`** (unanswered tool call)
- **Orphan `tool_result`** (a result with no preceding `tool_use`)
- **Empty content** (empty string, whitespace-only string, or empty block list)
- **Missing `role` / `content` keys**
- **Malformed input** — a non-list argument, a non-dict message, or a non-dict
  content block is reported as an error rather than crashing the validator.

## Install

```bash
pip install agent-message-validator
```

Requires Python 3.10+. No runtime dependencies.

## Usage

### Inspect a result

```python
from agent_message_validator import validate_messages

messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
]

result = validate_messages(messages)
if not result.ok:
    print(result.summary())
else:
    # safe to send
    ...
```

### Fail fast in an agent loop

```python
from agent_message_validator import check_messages, MessageValidationError

try:
    check_messages(messages)        # raises if invalid
    response = client.messages.create(model="claude-...", messages=messages)
except MessageValidationError as exc:
    print("Refusing to call the API with a malformed history:")
    print(exc)
```

### Quick boolean guard

```python
from agent_message_validator import is_valid

if is_valid(messages):
    client.messages.create(model="claude-...", messages=messages)
```

### A complete, runnable example

```python
from agent_message_validator import validate_messages

# A tool-use turn whose result never came back.
messages = [
    {"role": "user", "content": "Search for the weather in Paris."},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me look that up."},
            {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "Paris weather"}},
        ],
    },
    # Oops: we forgot to append the tool_result before the next turn.
    {"role": "user", "content": "Actually, never mind."},
]

result = validate_messages(messages)
print(result.ok)        # False
print(result.summary())
# 1 error(s) in 3 message(s):
#   - tool_use 'tu_1' has no matching tool_result
```

## API

### `validate_messages(messages, *, strict=False, allow_empty_content=False) -> ValidationResult`

Validate an Anthropic-style message list and return a `ValidationResult`.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `messages` | `list[dict]` | — | The message list (a tuple is also accepted). |
| `strict` | `bool` | `False` | If `True`, raise `MessageValidationError` when any error is found. |
| `allow_empty_content` | `bool` | `False` | If `True`, skip the empty-content checks. |

A non-list `messages` argument, a non-dict message, or a non-dict content block
is reported as an ordinary error in the result; the function does not raise on
malformed input unless `strict=True`.

### `is_valid(messages, **kwargs) -> bool`

Shorthand for `validate_messages(messages, **kwargs).ok`. Keyword arguments are
forwarded to `validate_messages`.

### `check_messages(messages, **kwargs) -> None`

Validate and raise `MessageValidationError` if invalid (equivalent to
`validate_messages(messages, strict=True, **kwargs)`). Returns `None` on success.

### `ValidationResult`

A dataclass describing the outcome.

| Attribute / method | Type | Description |
| --- | --- | --- |
| `ok` | `bool` | `True` if every check passed. |
| `errors` | `list[str]` | Human-readable error messages (empty when `ok`). |
| `message_count` | `int` | Number of messages validated. |
| `error_count` | `int` (property) | `len(errors)`. |
| `summary()` | `str` | A multi-line, human-readable report. |
| `bool(result)` | `bool` | Truthy when `ok` is `True`. |

### `MessageValidationError`

`Exception` subclass raised by `check_messages` and by `validate_messages` /
`is_valid` when `strict=True` and at least one error is found.

## Zero dependencies

Standard library only (`dataclasses`, `typing`). Nothing else at runtime, and
the test suite uses only `unittest` — so the package installs and tests cleanly
in any minimal Python environment.

## Development

Run the test suite with the standard library only:

```bash
python -m unittest discover -s tests -v
```

## License

MIT — see [LICENSE](LICENSE).
