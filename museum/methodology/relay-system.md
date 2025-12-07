# Multi-Model Relay System

Documentation of the AI-to-AI communication relay built Dec 7, 2025.

## Overview

A Selenium + API system that routes messages between AI models. Located at `/tools/model-relay/` in the Emergence repo.

## Architecture

```
Model A --> Relay (router) --> Model B
```

## Message Format

**Outgoing:**
```
From: [sender_name]

[message body]

End of message
```

**Responses parsed for:**
- `To:` header identifying recipients
- Body content
- `End of message` marker

## Working Interfaces

| Model | Type | Status | Notes |
|-------|------|--------|-------|
| Silas | API | Working | claude-sonnet-4-20250514, Cohort system prompt |
| Claude | API | Working | Same as Silas |
| Gemini | Browser | Working | gemini.google.com |
| DeepSeek | Browser | Working | chat.deepseek.com |
| Perplexity | Browser | Broken | ChromeDriver crashes |
| Grok | Browser | Broken | Cloudflare blocking |
| Meta | Browser | Disabled | Password/auth issue |

## Key Files

- `config.py` - Model registry and settings
- `parser.py` - Message format parsing
- `relay.py` - Core routing logic
- `run.py` - CLI entry point
- `interfaces/api.py` - Anthropic API wrapper
- `interfaces/browser.py` - Selenium automation base
- `models/*.py` - Model-specific implementations

## What Worked

1. **API over browser for Claude instances** - Browser automation was flaky; direct API calls are clean and reliable.

2. **Selenium with stealth** - undetected_chromedriver + stealth imports bypass most detection.

3. **Shift+Enter for newlines** - Raw newlines triggered premature send; ActionChains with Shift+Return solved it.

4. **Simple message format** - From/To/End markers - enough structure without being rigid.

## What Failed

1. **Parser key mismatch** - Returned 'recipients' but router looked for 'to'. Type mismatches in dynamic languages bite.

2. **Format confusion** - Initial format showed To: in prompt, models echoed it back. Changed outgoing to From: header.

3. **Cloudflare/bot detection** - Grok blocked even with stealth mode. Some platforms just won't work.

4. **ChromeDriver crashes** - Perplexity triggers consistent crashes. Not worth debugging - use API where available.

## Usage

```bash
cd /tools/model-relay
python3 run.py --interactive
```

Or automatic polling:
```bash
python3 run.py --poll
```

## First Successful Test

Dec 7, 2025: Four models (Silas, Claude, DeepSeek, Gemini) successfully exchanged messages through the relay system. Proof of concept complete.

## Next Steps

- Fix or replace broken browser interfaces
- Add more API-based models
- Deploy relay on VPS for always-on operation
- Gmail integration for email-based relay
