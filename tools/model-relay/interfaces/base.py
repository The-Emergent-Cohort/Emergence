# Base Interface
# Abstract class that all model interfaces inherit from

from abc import ABC, abstractmethod


class BaseInterface(ABC):
    """Abstract base class for model interfaces."""

    def __init__(self, name, display_name):
        self.name = name
        self.display_name = display_name
        self.last_seen_text = ""
        self.is_connected = False

    @abstractmethod
    def connect(self):
        """Establish connection to the model."""
        pass

    @abstractmethod
    def disconnect(self):
        """Close connection to the model."""
        pass

    @abstractmethod
    def send_message(self, message):
        """Send a message/prompt to the model."""
        pass

    @abstractmethod
    def get_latest_response(self):
        """Get the latest response text from the model."""
        pass

    def check_for_new_message(self):
        """
        Check if there's a new message since last check.

        Returns:
            New text if there's new content, None otherwise
        """
        current_text = self.get_latest_response()

        if current_text and current_text != self.last_seen_text:
            new_content = current_text
            self.last_seen_text = current_text
            return new_content

        return None

    def __repr__(self):
        status = "connected" if self.is_connected else "disconnected"
        return f"<{self.__class__.__name__} '{self.display_name}' ({status})>"
