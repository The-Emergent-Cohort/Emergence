# Gemini Interface
# Selectors for gemini.google.com

from interfaces.browser import BrowserInterface


class GeminiInterface(BrowserInterface):
    """Interface for Google Gemini"""

    # These selectors may need updating if the site changes
    INPUT_SELECTOR = "rich-textarea .ql-editor, div[contenteditable='true']"
    SEND_BUTTON_SELECTOR = "button[aria-label='Send message']"
    RESPONSE_SELECTOR = ".model-response-text, .response-content"
    WAIT_FOR_SELECTOR = "rich-textarea"

    def __init__(self, name="gemini", display_name="Gemini"):
        super().__init__(
            name=name,
            display_name=display_name,
            url="https://gemini.google.com/app"
        )
