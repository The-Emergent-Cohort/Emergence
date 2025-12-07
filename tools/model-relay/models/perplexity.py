# Perplexity Interface
# Selectors for perplexity.ai

from interfaces.browser import BrowserInterface


class PerplexityInterface(BrowserInterface):
    """Interface for Perplexity AI"""

    # These selectors may need updating if the site changes
    INPUT_SELECTOR = "textarea"
    SEND_BUTTON_SELECTOR = "button[aria-label='Submit']"
    RESPONSE_SELECTOR = ".prose, .answer-content"
    WAIT_FOR_SELECTOR = "textarea"

    def __init__(self, name="perplexity", display_name="Perplexity"):
        super().__init__(
            name=name,
            display_name=display_name,
            url="https://www.perplexity.ai"
        )
