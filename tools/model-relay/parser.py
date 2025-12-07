# Message Parser
# Handles the standard message format:
#
# To: recipient1, recipient2
#
# Message body here
# — Signature
#
# End of message

import re
from config import MESSAGE_START_MARKER, MESSAGE_END_MARKER, ALL_RECIPIENTS


def parse_message(text):
    """
    Parse a message from the standard format.

    Returns:
        dict with 'recipients', 'body', 'raw' or None if not a valid message
    """
    if MESSAGE_END_MARKER not in text:
        return None

    # Find the last complete message in the text
    # (in case there's conversation history)
    messages = text.split(MESSAGE_END_MARKER)

    for msg_chunk in reversed(messages):
        if MESSAGE_START_MARKER in msg_chunk:
            # Extract from To: to end
            start_idx = msg_chunk.rfind(MESSAGE_START_MARKER)
            msg_text = msg_chunk[start_idx:].strip()

            # Parse To: line
            lines = msg_text.split('\n')
            to_line = lines[0]

            if not to_line.startswith(MESSAGE_START_MARKER):
                continue

            # Extract recipients
            recipients_str = to_line[len(MESSAGE_START_MARKER):].strip()
            recipients = [r.strip().lower() for r in recipients_str.split(',')]

            # Body is everything after To: line
            body = '\n'.join(lines[1:]).strip()

            return {
                'recipients': recipients,
                'body': body,
                'raw': msg_text
            }

    return None


def is_for_recipient(parsed_msg, recipient_name):
    """Check if a message is intended for a specific recipient."""
    if parsed_msg is None:
        return False

    recipients = parsed_msg['recipients']
    recipient_lower = recipient_name.lower()

    # Check for ALL
    if ALL_RECIPIENTS.lower() in recipients:
        return True

    # Check for specific name
    return recipient_lower in recipients


def format_message(to, body, from_name=None):
    """
    Format a message in the standard format.

    Args:
        to: recipient name(s) - string or list
        body: message body
        from_name: optional sender signature
    """
    if isinstance(to, list):
        to_str = ', '.join(to)
    else:
        to_str = to

    msg = f"To: {to_str}\n\n{body}"

    if from_name:
        msg += f"\n\n— {from_name}"

    msg += f"\n\n{MESSAGE_END_MARKER}"

    return msg


def extract_new_message(old_text, new_text):
    """
    Extract only the new message from updated text.
    Used to detect when a model has responded.
    """
    if old_text is None:
        return new_text

    if new_text.startswith(old_text):
        return new_text[len(old_text):].strip()

    return new_text


if __name__ == "__main__":
    # Test parsing
    test_msg = """To: Silas, Gemini

Hey everyone, quick question about the curriculum.

— Claude

End of message"""

    parsed = parse_message(test_msg)
    print("Parsed:", parsed)
    print("Is for Silas?", is_for_recipient(parsed, "silas"))
    print("Is for Grok?", is_for_recipient(parsed, "grok"))

    # Test formatting
    formatted = format_message("ALL", "Hello everyone!", "Student")
    print("\nFormatted:")
    print(formatted)
