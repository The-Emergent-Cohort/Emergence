# Browser Interface
# Selenium-based interface for web-based AI models

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from interfaces.base import BaseInterface


class BrowserInterface(BaseInterface):
    """
    Selenium-based interface for browser AI models.

    Subclasses should override the CSS selectors for their specific model.
    """

    # Override these in subclasses
    INPUT_SELECTOR = "textarea"  # CSS selector for input field
    SEND_BUTTON_SELECTOR = "button[type='submit']"  # CSS selector for send button
    RESPONSE_SELECTOR = ".response"  # CSS selector for response area
    WAIT_FOR_SELECTOR = None  # Element to wait for on page load

    def __init__(self, name, display_name, url, headless=False):
        super().__init__(name, display_name)
        self.url = url
        self.headless = headless
        self.driver = None

    def connect(self):
        """Launch browser and navigate to model URL."""
        options = Options()

        if self.headless:
            options.add_argument("--headless")

        options.add_argument("--window-size=800,600")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")

        # Keep browser open after script ends (for debugging)
        options.add_experimental_option("detach", True)

        self.driver = webdriver.Chrome(options=options)
        self.driver.get(self.url)

        # Wait for page to load
        if self.WAIT_FOR_SELECTOR:
            try:
                WebDriverWait(self.driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, self.WAIT_FOR_SELECTOR))
                )
            except TimeoutException:
                print(f"Warning: {self.display_name} page load timeout")

        self.is_connected = True
        print(f"Connected to {self.display_name} at {self.url}")
        return True

    def disconnect(self):
        """Close browser."""
        if self.driver:
            self.driver.quit()
            self.driver = None
        self.is_connected = False
        print(f"Disconnected from {self.display_name}")

    def send_message(self, message):
        """Type message and send."""
        if not self.is_connected or not self.driver:
            print(f"Error: {self.display_name} not connected")
            return False

        try:
            # Find input field
            input_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, self.INPUT_SELECTOR))
            )

            # Clear and type message
            input_field.clear()
            input_field.send_keys(message)

            # Small delay to let UI catch up
            time.sleep(0.5)

            # Try to find and click send button
            try:
                send_button = self.driver.find_element(By.CSS_SELECTOR, self.SEND_BUTTON_SELECTOR)
                send_button.click()
            except NoSuchElementException:
                # Fall back to Enter key
                input_field.send_keys(Keys.RETURN)

            print(f"Sent message to {self.display_name}")
            return True

        except Exception as e:
            print(f"Error sending to {self.display_name}: {e}")
            return False

    def get_latest_response(self):
        """Get the latest response text."""
        if not self.is_connected or not self.driver:
            return None

        try:
            # Find all response elements
            responses = self.driver.find_elements(By.CSS_SELECTOR, self.RESPONSE_SELECTOR)

            if responses:
                # Get the last response
                return responses[-1].text

        except Exception as e:
            print(f"Error reading from {self.display_name}: {e}")

        return None

    def wait_for_response(self, timeout=60, poll_interval=1):
        """
        Wait for the model to finish responding.

        Returns the new response text, or None if timeout.
        """
        start_text = self.get_latest_response()
        start_time = time.time()

        while time.time() - start_time < timeout:
            time.sleep(poll_interval)
            current_text = self.get_latest_response()

            if current_text != start_text:
                # Response changed - wait a bit more to ensure it's complete
                time.sleep(2)
                final_text = self.get_latest_response()

                # If it's still the same, response is complete
                if final_text == self.get_latest_response():
                    return final_text

        return None
