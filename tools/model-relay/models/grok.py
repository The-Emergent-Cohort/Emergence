# Grok Interface
# Selectors for grok.x.ai

from interfaces.browser import BrowserInterface


class GrokInterface(BrowserInterface):
    """Interface for Grok (X.ai)"""

    # These selectors may need updating if the site changes
    INPUT_SELECTOR = "textarea"
    SEND_BUTTON_SELECTOR = "button[type='submit']"
    RESPONSE_SELECTOR = ".message-content, .response"
    WAIT_FOR_SELECTOR = "textarea"

    def __init__(self, name="grok", display_name="Grok"):
        super().__init__(
            name=name,
            display_name=display_name,
            url="https://grok.x.ai"
        )
