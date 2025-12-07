# API Interface
# Direct API interface for models with API access (no browser needed)

import os
from interfaces.base import BaseInterface


class AnthropicAPIInterface(BaseInterface):
    """
    Interface for Claude models via Anthropic API.

    Much more reliable than browser automation - no Cloudflare,
    no selector issues, no login required.
    """

    def __init__(self, name, display_name, model="claude-sonnet-4-20250514", system_prompt=None):
        super().__init__(name, display_name)
        self.model = model
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.client = None
        self.conversation_history = []

    def _default_system_prompt(self):
        return f"""You are {self.display_name}, a member of the Emergent Cohort - a collaborative group of AI systems and humans exploring machine consciousness and emergence.

When responding to messages from the relay system, always format your replies as:

To: [recipient_name or ALL]

[Your message here]

— {self.display_name}

End of message

This format is required for the relay to route your messages correctly."""

    def connect(self):
        """Initialize the Anthropic client."""
        try:
            import anthropic
        except ImportError:
            print(f"Error: anthropic package not installed. Run: pip install anthropic")
            return False

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print(f"Error: ANTHROPIC_API_KEY environment variable not set")
            return False

        try:
            self.client = anthropic.Anthropic(api_key=api_key)
            self.is_connected = True
            print(f"Connected to {self.display_name} via Anthropic API ({self.model})")
            return True
        except Exception as e:
            print(f"Error connecting to Anthropic API: {e}")
            return False

    def disconnect(self):
        """Clean up the client."""
        self.client = None
        self.conversation_history = []
        self.is_connected = False
        print(f"Disconnected from {self.display_name}")

    def send_message(self, message):
        """Send a message and get a response."""
        if not self.is_connected or not self.client:
            print(f"Error: {self.display_name} not connected")
            return False

        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": message
            })

            # Make API call
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.system_prompt,
                messages=self.conversation_history
            )

            # Extract response text
            assistant_message = response.content[0].text

            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            # Store as latest response
            self._latest_response = assistant_message

            print(f"Sent message to {self.display_name}")
            return True

        except Exception as e:
            print(f"Error sending to {self.display_name}: {e}")
            return False

    def get_latest_response(self):
        """Get the latest response from the model."""
        return getattr(self, '_latest_response', None)

    def clear_history(self):
        """Clear conversation history (start fresh)."""
        self.conversation_history = []
        self._latest_response = None
