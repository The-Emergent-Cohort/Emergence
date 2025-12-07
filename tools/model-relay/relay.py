# Model Relay
# Routes messages between AI models using browser automation

import time
from typing import Dict, Optional, List
from config import MODELS, POLL_INTERVAL, DEFAULT_TIMEOUT
from parser import parse_message, is_for_recipient, format_message
from models import MODEL_CLASSES


class ModelRelay:
    """
    Main relay that manages connections to multiple AI models
    and routes messages between them.
    """

    def __init__(self):
        self.interfaces: Dict[str, any] = {}
        self.last_responses: Dict[str, str] = {}
        self.running = False

    def initialize(self, models: Optional[List[str]] = None):
        """
        Initialize connections to specified models.
        If models is None, initializes all enabled models from config.
        """
        target_models = models or [
            name for name, cfg in MODELS.items()
            if cfg.get("enabled", True)
        ]

        print(f"Initializing models: {target_models}")

        for name in target_models:
            if name not in MODELS:
                print(f"  Warning: Unknown model '{name}', skipping")
                continue

            cfg = MODELS[name]
            if not cfg.get("enabled", True):
                print(f"  {name}: disabled, skipping")
                continue

            model_type = cfg.get("type", "browser")

            if model_type == "browser":
                if name not in MODEL_CLASSES:
                    print(f"  Warning: No interface class for '{name}'")
                    continue

                try:
                    interface = MODEL_CLASSES[name](name=name)
                    interface.connect()
                    self.interfaces[name] = interface
                    self.last_responses[name] = ""
                    print(f"  {name}: connected")
                except Exception as e:
                    print(f"  {name}: failed to connect - {e}")

            elif model_type == "local":
                print(f"  {name}: local model support not yet implemented")

        print(f"Initialized {len(self.interfaces)} models")

    def shutdown(self):
        """Disconnect from all models."""
        print("Shutting down relay...")
        for name, interface in self.interfaces.items():
            try:
                interface.disconnect()
                print(f"  {name}: disconnected")
            except Exception as e:
                print(f"  {name}: error disconnecting - {e}")
        self.interfaces.clear()
        self.running = False

    def send_to(self, recipient: str, message: str, from_name: str = "relay"):
        """Send a message to a specific model."""
        if recipient.lower() == "all":
            # Send to all connected models
            for name in self.interfaces:
                self._send_single(name, message, from_name)
        elif recipient in self.interfaces:
            self._send_single(recipient, message, from_name)
        else:
            print(f"Unknown recipient: {recipient}")

    def _send_single(self, name: str, message: str, from_name: str):
        """Send message to a single model."""
        interface = self.interfaces.get(name)
        if not interface:
            return

        try:
            formatted = format_message(to=name, body=message, from_name=from_name)
            interface.send_message(formatted)
            print(f"[-> {name}] Message sent ({len(message)} chars)")
        except Exception as e:
            print(f"[-> {name}] Failed to send: {e}")

    def check_for_messages(self) -> List[dict]:
        """
        Poll all models for new responses.
        Returns list of new messages with source and content.
        """
        new_messages = []

        for name, interface in self.interfaces.items():
            try:
                response = interface.get_latest_response()
                if response and response != self.last_responses.get(name, ""):
                    self.last_responses[name] = response
                    new_messages.append({
                        "from": name,
                        "content": response,
                        "timestamp": time.time()
                    })
            except Exception as e:
                print(f"[{name}] Error checking for messages: {e}")

        return new_messages

    def route_message(self, message: dict):
        """
        Parse a message and route it to the intended recipients.
        """
        content = message.get("content", "")
        from_name = message.get("from", "unknown")

        parsed = parse_message(content)
        if not parsed:
            # Not in standard format, don't route
            print(f"[{from_name}] Response not in routing format, skipping")
            return

        recipients = parsed.get("to", [])
        body = parsed.get("body", "")

        if not recipients or not body:
            return

        print(f"[{from_name}] Routing to: {recipients}")

        for recipient in recipients:
            recipient_lower = recipient.lower()

            # Don't send back to sender
            if recipient_lower == from_name.lower():
                continue

            # Handle special recipients
            if recipient_lower in ("b", "human", "operator"):
                print(f"[{from_name} -> Human]: {body[:100]}...")
                # Could trigger notification or log for human attention
                continue

            if recipient_lower == "all":
                for target in self.interfaces:
                    if target != from_name:
                        self._send_single(target, body, from_name)
            elif recipient_lower in self.interfaces:
                self._send_single(recipient_lower, body, from_name)
            else:
                print(f"[{from_name}] Unknown recipient: {recipient}")

    def run_loop(self, poll_interval: float = None):
        """
        Main loop: poll for messages and route them.
        Press Ctrl+C to stop.
        """
        interval = poll_interval or POLL_INTERVAL
        self.running = True
        print(f"\nRelay running. Polling every {interval}s. Press Ctrl+C to stop.\n")

        try:
            while self.running:
                new_messages = self.check_for_messages()

                for msg in new_messages:
                    print(f"\n[{msg['from']}] New message detected")
                    self.route_message(msg)

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.shutdown()


def create_relay(models: Optional[List[str]] = None) -> ModelRelay:
    """Factory function to create and initialize a relay."""
    relay = ModelRelay()
    relay.initialize(models)
    return relay
