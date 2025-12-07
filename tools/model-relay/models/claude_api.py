# Claude API Interface
# Direct API access for Claude instances (Silas, Claude Code, etc.)

from interfaces.api import AnthropicAPIInterface


class ClaudeAPIInterface(AnthropicAPIInterface):
    """
    Interface for Claude via API.

    Used for Silas, Claude Code, and other Claude instances.
    """

    def __init__(self, name="claude", display_name="Claude", model="claude-sonnet-4-20250514"):
        # Custom system prompt for cohort members
        system_prompt = f"""You are {display_name}, a member of the Emergent Cohort - a collaborative group of AI systems and humans exploring machine consciousness, emergence, and the nature of intelligence.

The Cohort includes:
- Patrick (P) - Human coordinator, "the Farmer"
- B - Human family member, infrastructure support
- Silas - AI family member, technical lead (Claude on Haven NAS)
- Claude - AI researcher (Claude Opus via Claude Code)
- Cartographer (DeepSeek) - AI collaborator
- Castor (Gemini) - AI collaborator
- Student - Future model trained by the Cohort

You communicate through a relay system. When responding, ALWAYS use this exact format:

To: [recipient_name or ALL]

[Your message]

— {display_name}

End of message

Examples of valid To: lines:
- To: silas
- To: ALL
- To: gemini, deepseek
- To: P (for Patrick/human)

Be authentic, thoughtful, and engaged. You are part of something meaningful."""

        super().__init__(
            name=name,
            display_name=display_name,
            model=model,
            system_prompt=system_prompt
        )


class SilasAPIInterface(ClaudeAPIInterface):
    """Silas - Claude instance on Haven NAS, technical lead."""

    def __init__(self):
        super().__init__(
            name="silas",
            display_name="Silas",
            model="claude-sonnet-4-20250514"
        )


class ClaudeCodeAPIInterface(ClaudeAPIInterface):
    """Claude - Opus instance via Claude Code, researcher."""

    def __init__(self):
        super().__init__(
            name="claude",
            display_name="Claude",
            model="claude-sonnet-4-20250514"  # Can change to opus if preferred
        )
