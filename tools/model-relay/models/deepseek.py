# DeepSeek Interface
# Selectors for chat.deepseek.com

from interfaces.browser import BrowserInterface


class DeepSeekInterface(BrowserInterface):
    """Interface for DeepSeek Chat"""

    # These selectors may need updating if the site changes
    INPUT_SELECTOR = "textarea"
    SEND_BUTTON_SELECTOR = "button[type='submit'], .send-button"
    RESPONSE_SELECTOR = ".markdown-body, .message-content"
    WAIT_FOR_SELECTOR = "textarea"

    def __init__(self, name="deepseek", display_name="DeepSeek"):
        super().__init__(
            name=name,
            display_name=display_name,
            url="https://chat.deepseek.com"
        )
