# Model Relay Configuration

# How often to check for new messages (seconds)
POLL_INTERVAL = 2

# Default timeout for waiting on responses (seconds)
DEFAULT_TIMEOUT = 60

# Message format markers
MESSAGE_START_MARKER = "To:"
MESSAGE_END_MARKER = "End of message"

# Special recipients
ALL_RECIPIENTS = "ALL"

# Model registry - add new models here
MODELS = {
    "silas": {
        "type": "browser",
        "url": "https://claude.ai",
        "display_name": "Silas",
        "enabled": True
    },
    "claude": {
        "type": "browser",
        "url": "https://claude.ai/code",
        "display_name": "Claude",
        "enabled": True
    },
    "gemini": {
        "type": "browser",
        "url": "https://gemini.google.com",
        "display_name": "Gemini",
        "enabled": True
    },
    "deepseek": {
        "type": "browser",
        "url": "https://chat.deepseek.com",
        "display_name": "DeepSeek",
        "enabled": True
    },
    "grok": {
        "type": "browser",
        "url": "https://grok.x.ai",
        "display_name": "Grok",
        "enabled": False  # Cloudflare blocking - disabled for now
    },
    "perplexity": {
        "type": "browser",
        "url": "https://perplexity.ai",
        "display_name": "Perplexity",
        "enabled": True
    },
    "meta": {
        "type": "browser",
        "url": "https://meta.ai",
        "display_name": "Meta",
        "enabled": True
    },
    "student": {
        "type": "local",
        "model_path": None,  # Set when model is trained
        "display_name": "Student",
        "enabled": False  # Enable when ready
    }
}

# Browser settings
BROWSER_SETTINGS = {
    "headless": False,  # Set True to hide windows
    "window_width": 800,
    "window_height": 600,
    # Stealth options to avoid bot detection
    "stealth_mode": True
}

# Chrome arguments to appear more human
CHROME_STEALTH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
    "--disable-dev-shm-usage",
    "--no-sandbox",
    "--disable-gpu",
    "--window-size=1920,1080",
    "--start-maximized",
    "--disable-extensions",
]

# Use existing Chrome profile (keeps logins, looks more legitimate)
# Set to your Chrome profile path, or None to use fresh profile
CHROME_PROFILE_PATH = None  # e.g., "/home/user/.config/google-chrome/Default"
