# Model Interfaces
# Export all model-specific interfaces

from models.claude_web import ClaudeWebInterface
from models.gemini import GeminiInterface
from models.deepseek import DeepSeekInterface
from models.grok import GrokInterface
from models.perplexity import PerplexityInterface

# Registry of available model classes
MODEL_CLASSES = {
    "silas": ClaudeWebInterface,
    "gemini": GeminiInterface,
    "deepseek": DeepSeekInterface,
    "grok": GrokInterface,
    "perplexity": PerplexityInterface,
}

__all__ = [
    "ClaudeWebInterface",
    "GeminiInterface",
    "DeepSeekInterface",
    "GrokInterface",
    "PerplexityInterface",
    "MODEL_CLASSES",
]
