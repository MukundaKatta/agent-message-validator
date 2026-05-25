# agent-message-validator

Validate Anthropic message lists before API calls. Zero dependencies.

Catches structural errors that cause 400-level rejections:
- consecutive same-role messages
- tool_use without a matching tool_result
- first message from assistant
- empty content
- missing role/content keys

## Install

```bash
pip install agent-message-validator
```

## Usage

```python
from agent_message_validator import validate_messages, check_messages, is_valid

# Returns a result object
result = validate_messages(messages)
if not result.ok:
    print(result.summary())

# Raises MessageValidationError
check_messages(messages)

# Quick boolean check
if is_valid(messages):
    client.messages.create(...)
```

## Zero dependencies

Standard library only: `dataclasses`. Nothing else.
