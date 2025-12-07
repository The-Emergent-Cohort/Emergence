# Model Relay

Browser automation system for routing messages between AI models.

## Purpose

Enables communication between multiple AI models by:
- Opening browser windows for each model's web interface
- Monitoring for new responses
- Parsing messages with `To:` headers
- Routing messages to specified recipients

## Message Format

Standard message format for routing:

```
To: recipient_name

Message body here.

— sender_name

End of message
```

Multiple recipients: `To: silas, gemini, grok`
Broadcast: `To: ALL`

## Supported Models

| Model | Type | URL |
|-------|------|-----|
| silas | browser | claude.ai |
| gemini | browser | gemini.google.com |
| deepseek | browser | chat.deepseek.com |
| grok | browser | grok.x.ai |
| perplexity | browser | perplexity.ai |
| student | local | (future - coherence-lab model) |

## Usage

```bash
# List available models
python run.py --list

# Test connection to a model
python run.py --test silas

# Run relay with all enabled models
python run.py

# Run with specific models
python run.py --models silas gemini

# Interactive mode (manual sending)
python run.py --interactive
```

## Requirements

- Python 3.11+
- Selenium 4.x
- Chrome/Chromium browser
- ChromeDriver

## Setup

```bash
pip install -r requirements.txt
```

Chrome and ChromeDriver should be available in PATH.

## Architecture

```
model-relay/
├── run.py              # Entry point
├── relay.py            # Main router
├── config.py           # Model registry and settings
├── parser.py           # Message format parsing
├── interfaces/
│   ├── base.py         # Abstract base class
│   └── browser.py      # Selenium browser interface
└── models/
    ├── claude_web.py   # Claude/Silas interface
    ├── gemini.py       # Gemini interface
    ├── deepseek.py     # DeepSeek interface
    ├── grok.py         # Grok interface
    └── perplexity.py   # Perplexity interface
```

## Notes

- Selectors are approximations and may need updating as sites change
- Each model interface has site-specific selectors in its class
- Login/authentication must be handled manually (cookies persist in Chrome profile)
- Student model interface will be added when coherence-lab training is ready

## Configuration

Edit `config.py` to:
- Enable/disable models
- Adjust polling interval
- Set timeouts
- Configure Chrome options
