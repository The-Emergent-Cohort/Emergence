# Meta AI Interface
# Browser interface for Meta AI (meta.ai)

from interfaces.browser import BrowserInterface


class MetaInterface(BrowserInterface):
    """
    Interface for Meta AI.

    Meta AI uses a chat-style interface.
    """

    # Meta AI selectors (may need adjustment)
    INPUT_SELECTOR = "textarea[placeholder*='message'], div[contenteditable='true']"
    SEND_BUTTON_SELECTOR = "button[aria-label='Send'], button[type='submit']"
    RESPONSE_SELECTOR = "[data-testid='message-content'], .message-content, div[class*='response']"
    WAIT_FOR_SELECTOR = "textarea, div[contenteditable='true']"

    def __init__(self, name="meta", display_name="Meta", url="https://meta.ai", headless=False):
        super().__init__(name, display_name, url, headless)

    def send_message(self, message):
        """
        Send message to Meta AI.
        """
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.keys import Keys
        import time

        if not self.is_connected or not self.driver:
            print(f"Error: {self.display_name} not connected")
            return False

        try:
            # Find input field
            input_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, self.INPUT_SELECTOR))
            )

            # Check if contenteditable div or textarea
            tag_name = input_field.tag_name.lower()

            if tag_name == "div" or input_field.get_attribute("contenteditable") == "true":
                # Use JavaScript for contenteditable
                self.driver.execute_script("arguments[0].innerText = arguments[1]", input_field, message)
                input_field.click()
            else:
                # Regular textarea
                input_field.clear()
                input_field.send_keys(message)

            time.sleep(0.5)

            # Try to find and click send button
            try:
                send_button = self.driver.find_element(By.CSS_SELECTOR, self.SEND_BUTTON_SELECTOR)
                send_button.click()
            except:
                # Fall back to Enter key
                input_field.send_keys(Keys.RETURN)

            print(f"Sent message to {self.display_name}")
            return True

        except Exception as e:
            print(f"Error sending to {self.display_name}: {e}")
            return False
