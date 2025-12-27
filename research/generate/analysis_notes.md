# generate() Function Analysis

## Sources to examine:
1. HuggingFace transformers (Python - most accessible)
2. llama.cpp (C++ - what Ollama uses)
3. Model-specific overrides

## Key questions:
- What control points exist?
- Where are constraints injected?
- What can be modified without recompiling?
