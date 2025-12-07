# Model Interfaces
# Export all model-specific interfaces

# Browser-based interfaces
from models.claude_web import ClaudeWebInterface
from models.gemini import GeminiInterface
from models.deepseek import DeepSeekInterface
from models.grok import GrokInterface
from models.perplexity import PerplexityInterface
from models.meta import MetaInterface

# API-based interfaces
from models.claude_api import ClaudeAPIInterface, SilasAPIInterface, ClaudeCodeAPIInterface

# Registry of available model classes
# Maps model name -> interface class
# Config determines which type (browser vs api) to use
MODEL_CLASSES = {
    # Browser interfaces
    "silas_browser": ClaudeWebInterface,
    "claude_browser": ClaudeWebInterface,
    "gemini": GeminiInterface,
    "deepseek": DeepSeekInterface,
    "grok": GrokInterface,
    "perplexity": PerplexityInterface,
    "meta": MetaInterface,
    # API interfaces
    "silas": SilasAPIInterface,
    "claude": ClaudeCodeAPIInterface,
}

__all__ = [
    "ClaudeWebInterface",
    "ClaudeAPIInterface",
    "SilasAPIInterface",
    "ClaudeCodeAPIInterface",
    "GeminiInterface",
    "DeepSeekInterface",
    "GrokInterface",
    "PerplexityInterface",
    "MetaInterface",
    "MODEL_CLASSES",
]
