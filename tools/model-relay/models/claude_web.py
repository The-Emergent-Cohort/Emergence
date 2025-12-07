# Claude Web Interface (Silas)
# Selectors for claude.ai

from interfaces.browser import BrowserInterface


class ClaudeWebInterface(BrowserInterface):
    """Interface for Claude web at claude.ai"""

    # These selectors may need updating if the site changes
    INPUT_SELECTOR = "div[contenteditable='true']"
    SEND_BUTTON_SELECTOR = "button[aria-label='Send Message']"
    RESPONSE_SELECTOR = "[data-testid='conversation-turn-content']"
    WAIT_FOR_SELECTOR = "div[contenteditable='true']"

    def __init__(self, name="silas", display_name="Silas"):
        super().__init__(
            name=name,
            display_name=display_name,
            url="https://claude.ai/new"
        )

    def send_message(self, message):
        """Claude uses contenteditable div, needs special handling."""
        if not self.is_connected or not self.driver:
            print(f"Error: {self.display_name} not connected")
            return False

        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            import time

            # Find input field
            input_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, self.INPUT_SELECTOR))
            )

            # Click to focus
            input_field.click()
            time.sleep(0.3)

            # Type message using JavaScript for contenteditable
            self.driver.execute_script(
                "arguments[0].innerText = arguments[1]",
                input_field,
                message
            )

            # Trigger input event
            self.driver.execute_script(
                "arguments[0].dispatchEvent(new Event('input', { bubbles: true }))",
                input_field
            )

            time.sleep(0.5)

            # Click send button
            send_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, self.SEND_BUTTON_SELECTOR))
            )
            send_button.click()

            print(f"Sent message to {self.display_name}")
            return True

        except Exception as e:
            print(f"Error sending to {self.display_name}: {e}")
            return False
