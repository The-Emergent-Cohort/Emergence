# Model Relay Configuration

# How often to check for new messages (seconds)
POLL_INTERVAL = 2

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
        "enabled": True
    },
    "perplexity": {
        "type": "browser",
        "url": "https://perplexity.ai",
        "display_name": "Perplexity",
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
    "window_height": 600
}
