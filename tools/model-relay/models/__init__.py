# Model Interfaces
# Export all model-specific interfaces

from models.claude_web import ClaudeWebInterface
from models.gemini import GeminiInterface
from models.deepseek import DeepSeekInterface
from models.grok import GrokInterface
from models.perplexity import PerplexityInterface
from models.meta import MetaInterface

# Registry of available model classes
MODEL_CLASSES = {
    "silas": ClaudeWebInterface,
    "claude": ClaudeWebInterface,  # Claude Code window
    "gemini": GeminiInterface,
    "deepseek": DeepSeekInterface,
    "grok": GrokInterface,
    "perplexity": PerplexityInterface,
    "meta": MetaInterface,
}

__all__ = [
    "ClaudeWebInterface",
    "GeminiInterface",
    "DeepSeekInterface",
    "GrokInterface",
    "PerplexityInterface",
    "MetaInterface",
    "MODEL_CLASSES",
]
