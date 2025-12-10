"""
Kaggle Notebook: Curriculum Format Design with Merlyn
======================================================

SETUP INSTRUCTIONS:
1. Create new Kaggle notebook
2. Turn on GPU (Settings → Accelerator → GPU P100)
3. Copy this entire file into a code cell
4. Run!

Or split into cells at the # %% markers for interactive use.
"""

# %% [markdown]
# # Curriculum Compiler: Format Design
#
# Using Merlyn Education Teacher Assistant to propose a curriculum format
# for developmental DI training.

# %% Install dependencies
!pip install -q transformers accelerate bitsandbytes datasets

# %% Imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import json

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% Load Merlyn (8-bit quantized)
print("\nLoading Merlyn Education Teacher Assistant (8-bit)...")
print("This may take a few minutes on first run...")

model_name = "MerlynMind/merlyn-education-teacher-assistant"

# 8-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print("Model loaded successfully!")
print(f"Model device: {next(model.parameters()).device}")

# %% Helper function for generation
def ask_merlyn(prompt, max_new_tokens=1024, temperature=0.7):
    """Generate response from Merlyn."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from response
    response = response[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
    return response.strip()

# %% Context about our project
PROJECT_CONTEXT = """
You are helping design a curriculum format for training a developmental AI system.

BACKGROUND:
- We're building a "Digital Intelligence" (DI) that learns like a child, not through brute force
- The student model has four components: Self (metacognition), Other (theory of mind),
  World (reality grounding), Temporal (memory/continuity)
- We use approval-seeking like children do with teachers
- Current training uses synthetic patterns (counting, alternating, fibonacci, etc.)

THE PROBLEM:
- We assumed AI would know how to teach AI - wrong assumption
- AI was trained by brute force, doesn't know how to LEARN
- New direction: use existing human pedagogy (early childhood education, 2E methods)

PEDAGOGICAL PRINCIPLES WE WANT TO APPLY:
- CPA (Concrete → Pictorial → Abstract) from Singapore Math
- I Do / We Do / You Do gradual release
- Interleaving (mix topics for better retention)
- Worked examples before practice problems
- Mastery before progression

SUBJECTS TO SUPPORT:
- Math (counting, arithmetic, word problems)
- Reading (letters, phonics, comprehension)
- Physics (motion, forces - for grounding)
- Music (rhythm, patterns)
- Art (shapes, strokes)

Each subject has different native representations. Math uses numbers, reading uses words,
physics uses positions/velocities, etc.
"""

# %% Ask Merlyn to propose format
FORMAT_PROMPT = PROJECT_CONTEXT + """

TASK:
Please propose a JSON format for curriculum items that:
1. Works across multiple subjects (math, reading, physics, etc.)
2. Preserves subject-native representations (don't flatten everything to text)
3. Includes pedagogical metadata (CPA stage, I Do/We Do/You Do phase, difficulty)
4. Supports scaffolding (hints, worked examples, corrections)
5. Tracks prerequisites and connections between topics

Provide:
1. The proposed JSON schema with explanations
2. Three example curriculum items (one math, one reading, one physics)
3. Any concerns or alternatives you'd suggest

Be specific and practical - this will be implemented in code.
"""

print("Asking Merlyn to propose curriculum format...")
print("=" * 60)
format_response = ask_merlyn(FORMAT_PROMPT, max_new_tokens=2048)
print(format_response)
print("=" * 60)

# %% Save the response
with open("merlyn_format_proposal.txt", "w") as f:
    f.write("PROMPT:\n")
    f.write(FORMAT_PROMPT)
    f.write("\n\nRESPONSE:\n")
    f.write(format_response)
print("\nSaved to merlyn_format_proposal.txt")

# %% Test with a real GSM8K problem
print("\n" + "=" * 60)
print("Testing with a real math problem from GSM8K...")
print("=" * 60)

# Load a sample from GSM8K
gsm8k = load_dataset("openai/gsm8k", "main", split="train[:5]")
sample = gsm8k[0]

COMPILE_PROMPT = PROJECT_CONTEXT + f"""

TASK:
Convert this grade school math problem into a structured curriculum item
using the format you proposed above.

PROBLEM:
{sample['question']}

SOLUTION:
{sample['answer']}

Provide the JSON curriculum item with all fields filled in appropriately.
Include scaffolding (worked example, hints, correction feedback).
Identify what CPA stage and I Do/We Do/You Do phase this would be used in.
"""

print(f"\nOriginal problem:\n{sample['question'][:200]}...")
print("\nAsking Merlyn to compile this into curriculum format...")
compile_response = ask_merlyn(COMPILE_PROMPT, max_new_tokens=1024)
print("\n" + compile_response)

# %% Save compiled example
with open("merlyn_compiled_example.txt", "w") as f:
    f.write("ORIGINAL PROBLEM:\n")
    f.write(sample['question'])
    f.write("\n\nSOLUTION:\n")
    f.write(sample['answer'])
    f.write("\n\nCOMPILED FORMAT:\n")
    f.write(compile_response)
print("\nSaved to merlyn_compiled_example.txt")

# %% Summary
print("\n" + "=" * 60)
print("COMPLETE!")
print("=" * 60)
print("""
Files created:
- merlyn_format_proposal.txt : The proposed curriculum format
- merlyn_compiled_example.txt : A compiled GSM8K example

Next steps:
1. Review the proposed format
2. Iterate if needed (run more prompts)
3. Download the files
4. Implement the format in the training system
""")

# %% [Optional] Interactive mode - uncomment to chat with Merlyn
# while True:
#     user_input = input("\nYou: ")
#     if user_input.lower() in ['quit', 'exit', 'q']:
#         break
#     response = ask_merlyn(PROJECT_CONTEXT + "\n\nUser question: " + user_input)
#     print(f"\nMerlyn: {response}")
