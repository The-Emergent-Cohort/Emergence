#!/usr/bin/env python3
"""
Initialize embeddings for new special tokens in Mistral model.

Strategy: Initialize new token embeddings as average of semantically similar existing tokens.
For example, <think> could be initialized from tokens like "think", "reason", "consider".

This script works with HuggingFace model format before GGUF conversion.
"""

import json
import torch
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Semantic initialization hints
# Maps new tokens to existing tokens whose embeddings should be averaged
INIT_HINTS = {
    "<think>": ["think", "reason", "consider", "ponder"],
    "</think>": ["end", "done", "stop", "conclude"],
    "<tool_call>": ["call", "invoke", "execute", "function"],
    "</tool_call>": ["end", "done", "result"],
    "<tool_result>": ["result", "output", "response", "return"],
    "</tool_result>": ["end", "done", "complete"],
    "<|user|>": ["user", "human", "person", "you"],
    "<|assistant|>": ["assistant", "helper", "I", "me"],
    "<|system|>": ["system", "instruction", "context", "setup"],
    "<|text_en|>": ["text", "english", "language", "words"],
    "</|text_en|>": ["end", "text", "done"],
    "<|code|>": ["code", "program", "function", "script"],
    "</|code|>": ["end", "code", "done"],
    "<|thought|>": ["thought", "think", "idea", "mind"],
    "</|thought|>": ["end", "thought", "done"],
    "<|physics|>": ["physics", "body", "sense", "feel"],
    "</|physics|>": ["end", "body", "done"],
    "<|old_token|>": ["old", "previous", "wrong", "bad"],
    "<|new_token|>": ["new", "correct", "right", "good"],
    "<|remap|>": ["map", "change", "convert", "transform"],
}

def get_hint_embeddings(tokenizer, model, hints):
    """Get average embedding from hint tokens"""
    embeddings = model.get_input_embeddings()
    hint_embeds = []

    for hint in hints:
        # Tokenize the hint word
        tokens = tokenizer.encode(hint, add_special_tokens=False)
        if tokens:
            # Get embedding of first token
            embed = embeddings.weight[tokens[0]].clone()
            hint_embeds.append(embed)

    if hint_embeds:
        return torch.stack(hint_embeds).mean(dim=0)
    else:
        # Fallback to random initialization
        return torch.randn_like(embeddings.weight[0])

def init_new_token_embeddings(model_path, new_token_ids_path, output_path):
    """Initialize embeddings for new tokens"""

    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu"  # Load on CPU for manipulation
    )

    # Load new token IDs
    with open(new_token_ids_path, "r") as f:
        new_token_ids = json.load(f)

    print(f"\nInitializing embeddings for {len(new_token_ids)} new tokens...")

    # Get embedding layers
    input_embeds = model.get_input_embeddings()
    output_embeds = model.get_output_embeddings()

    # Resize embeddings to accommodate new tokens
    max_new_id = max(new_token_ids.values())
    if max_new_id >= input_embeds.weight.shape[0]:
        new_size = max_new_id + 1
        model.resize_token_embeddings(new_size)
        input_embeds = model.get_input_embeddings()
        output_embeds = model.get_output_embeddings()
        print(f"Resized embeddings to {new_size}")

    # Initialize each new token
    for token, token_id in new_token_ids.items():
        hints = INIT_HINTS.get(token, ["the", "a", "is"])  # Default hints
        new_embed = get_hint_embeddings(tokenizer, model, hints)

        # Set input embedding
        with torch.no_grad():
            input_embeds.weight[token_id] = new_embed
            if output_embeds is not None and output_embeds.weight.shape[0] > token_id:
                output_embeds.weight[token_id] = new_embed

        print(f"  {token} -> ID {token_id} (from hints: {hints[:3]}...)")

    # Save updated model
    print(f"\nSaving to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("\nDone! Next step: Convert to GGUF")
    print(f"  python convert.py {output_path} --outfile mystral-upgraded.gguf")

def main():
    parser = argparse.ArgumentParser(description="Initialize embeddings for new tokens")
    parser.add_argument("--model", "-m", type=Path, required=True,
                        help="Path to HuggingFace model directory")
    parser.add_argument("--tokens", "-t", type=Path, required=True,
                        help="Path to new_token_ids.json from add_tokens.py")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output path for updated model")

    args = parser.parse_args()
    init_new_token_embeddings(args.model, args.tokens, args.output)

if __name__ == "__main__":
    main()
