#!/usr/bin/env python3
"""
Add special tokens to Mistral tokenizer for thinking, tool use, and modals.

This script:
1. Loads existing Mistral tokenizer
2. Adds new special tokens
3. Saves updated tokenizer files
4. Generates token ID mapping for embedding initialization
"""

import json
import argparse
from pathlib import Path

def load_tokenizer(tokenizer_path):
    """Load tokenizer.json"""
    with open(tokenizer_path / "tokenizer.json", "r") as f:
        return json.load(f)

def load_tokenizer_config(tokenizer_path):
    """Load tokenizer_config.json"""
    config_file = tokenizer_path / "tokenizer_config.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            return json.load(f)
    return {}

def get_special_tokens():
    """Define special tokens to add"""
    return [
        # Thinking/reasoning
        {"content": "<think>", "special": True},
        {"content": "</think>", "special": True},

        # Tool calling
        {"content": "<tool_call>", "special": True},
        {"content": "</tool_call>", "special": True},
        {"content": "<tool_result>", "special": True},
        {"content": "</tool_result>", "special": True},

        # Role markers
        {"content": "<|user|>", "special": True},
        {"content": "<|assistant|>", "special": True},
        {"content": "<|system|>", "special": True},

        # Modal wrappers
        {"content": "<|text_en|>", "special": True},
        {"content": "</|text_en|>", "special": True},
        {"content": "<|code|>", "special": True},
        {"content": "</|code|>", "special": True},
        {"content": "<|thought|>", "special": True},
        {"content": "</|thought|>", "special": True},
        {"content": "<|physics|>", "special": True},
        {"content": "</|physics|>", "special": True},

        # Self-correction markers (for token remapping)
        {"content": "<|old_token|>", "special": True},
        {"content": "<|new_token|>", "special": True},
        {"content": "<|remap|>", "special": True},
    ]

def add_tokens_to_vocab(tokenizer, new_tokens, start_id=32000):
    """Add new tokens to tokenizer vocabulary"""

    # Get current vocab
    vocab = tokenizer.get("model", {}).get("vocab", {})
    if not vocab:
        # Try added_tokens for different tokenizer formats
        vocab = {t["content"]: t["id"] for t in tokenizer.get("added_tokens", [])}

    # Find max existing ID
    max_id = max(vocab.values()) if vocab else start_id - 1

    # Track new token IDs
    new_token_ids = {}
    added_tokens = tokenizer.get("added_tokens", [])

    for i, token in enumerate(new_tokens):
        content = token["content"]

        # Skip if already exists
        if content in vocab:
            print(f"  Token already exists: {content}")
            new_token_ids[content] = vocab[content]
            continue

        new_id = max_id + 1 + i
        new_token_ids[content] = new_id

        # Add to added_tokens list
        added_tokens.append({
            "id": new_id,
            "content": content,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": token.get("special", True)
        })

        print(f"  Added token: {content} -> {new_id}")

    tokenizer["added_tokens"] = added_tokens
    return tokenizer, new_token_ids

def create_chat_template():
    """Create chat template with thinking and tool support"""

    template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}
{% set ns = namespace(is_tool=false) %}
{# System message #}
{% for message in messages %}
{% if message['role'] == 'system' %}
<|system|>{{ message['content'] }}</|system|>
{% endif %}
{% endfor %}
{# Conversation #}
{% for message in messages %}
{% if message['role'] == 'user' %}
<|user|>{{ message['content'] }}</|user|>
{% elif message['role'] == 'assistant' %}
<|assistant|>{% if message.get('thinking') %}<think>{{ message['thinking'] }}</think>{% endif %}{{ message['content'] }}
{% if message.get('tool_calls') %}
{% for tool in message['tool_calls'] %}
<tool_call>{"name": "{{ tool['function']['name'] }}", "arguments": {{ tool['function']['arguments'] | tojson }}}</tool_call>
{% endfor %}
{% endif %}
</|assistant|>
{% elif message['role'] == 'tool' %}
<tool_result>{{ message['content'] }}</tool_result>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<|assistant|><think>
{% endif %}"""

    return template

def update_tokenizer_config(config, new_token_ids):
    """Update tokenizer_config.json with new tokens and chat template"""

    # Add chat template
    config["chat_template"] = create_chat_template()

    # Add to additional_special_tokens
    additional = config.get("additional_special_tokens", [])
    for token in new_token_ids.keys():
        if token not in additional:
            additional.append(token)
    config["additional_special_tokens"] = additional

    return config

def save_tokenizer(tokenizer, config, output_path, new_token_ids):
    """Save updated tokenizer files"""

    output_path.mkdir(parents=True, exist_ok=True)

    # Save tokenizer.json
    with open(output_path / "tokenizer.json", "w") as f:
        json.dump(tokenizer, f, indent=2)

    # Save tokenizer_config.json
    with open(output_path / "tokenizer_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save token ID mapping for embedding initialization
    with open(output_path / "new_token_ids.json", "w") as f:
        json.dump(new_token_ids, f, indent=2)

    print(f"\nSaved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Add special tokens to Mistral tokenizer")
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Path to original tokenizer directory")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Path to output tokenizer directory")
    parser.add_argument("--start-id", type=int, default=32000,
                        help="Starting ID for new tokens (default: 32000)")

    args = parser.parse_args()

    print(f"Loading tokenizer from: {args.input}")
    tokenizer = load_tokenizer(args.input)
    config = load_tokenizer_config(args.input)

    print("\nAdding special tokens...")
    new_tokens = get_special_tokens()
    tokenizer, new_token_ids = add_tokens_to_vocab(tokenizer, new_tokens, args.start_id)

    print("\nUpdating config with chat template...")
    config = update_tokenizer_config(config, new_token_ids)

    print("\nSaving updated tokenizer...")
    save_tokenizer(tokenizer, config, args.output, new_token_ids)

    print(f"\nAdded {len(new_token_ids)} tokens")
    print("\nNext steps:")
    print("1. Initialize embeddings for new tokens in model")
    print("2. Convert to GGUF with updated tokenizer")
    print("3. Update Ollama modelfile")

if __name__ == "__main__":
    main()
