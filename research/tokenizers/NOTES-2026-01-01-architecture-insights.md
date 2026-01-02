# Architecture Insights - 2026-01-01

Session notes from discussion with DI project lead.

## Parameter Counts Are Brute-Forced Index Tables

Current LLM "parameter count" is meaningless in traditional context:
- Fixed value established at model creation
- Exists solely to create a blank, fixed-length index table
- Maps token IDs to weight vectors
- Brute-forces a database function into writing byte-encoded data instead of proper DB entries

Our approach: The database IS the index. No need to predetermine size.

## Differential vs Full Calculation

Current LLMs: Full matrix multiplication every token, every forward pass. No deltas.

Structured approach: Differential calculation like game/physics engines.
- Only compute what changed
- GPU cycles focused on relevant data
- Same principle that lets complex games run at 60fps on consumer hardware

The bottleneck shifts from raw GPU power to CPU data serving:
- Can CPU serve data fast enough?
- Can it predict/prefetch what's needed next?
- Stutter comes from cache misses and data starvation, not GPU limits

## Hardware Spec: Data Structure First, Raw Power Second

How many parameters hardware can handle becomes a function of:
1. Data structure efficiency
2. Raw compute power (secondary)

Well-structured system on modest hardware can outperform brute-force blob on expensive hardware.
Same reason indexed DB query beats full table scan on slower machines.

"Parameter count" stops being a spec and becomes emergent property of workload.

## Deltas: Diffusion vs Clarity

Current LLMs: Deltas are diffusion
- Gradient updates smear information across billions of weights
- Knowledge blended, approximated, statistically distributed
- Can't point to where a fact lives

Structured system: Deltas are clarity
- Discrete entry says exactly what changed
- Relationships explicit, locatable, traceable
- Each delta sharpens rather than blurs

## Var Database

Separate database for working state / temporary data:
- Inference chains for recursion and batch analysis
- Physics data
- Temporary data structures
- Anything mutable and in-flight

Why separate:
- Different access patterns (write-heavy, temporal)
- Different lifecycle (created/destroyed per work block)
- Doesn't pollute stable concept/relationship data
- Can be indexed differently (recency, dependency chain)

## Everything Is Containers + Token References

All databases are just containers. Token label is just reference.
- Primitives, physics vectors, inference chains - all just entries
- Type distinction (concept vs var vs identity) is organizational, not structural
- Genomic notation doesn't care what it's pointing to

Data division is about efficient loading/unloading from RAM/VRAM.
Far more efficient to swap memory blocks of a few GB than hold monolithic file.

## Language DB: Common/Delta Structure

For large languages with variants (English US/GB/CA/AU):

### Dialect Numbering
- 0 = core (canonical reference, not a variant)
- 1+ = dialect deltas

### Core Behavior
- Contains full entries for shared/unambiguous tokens
- For divergent tokens: keeps token + references to which deltas have meanings

Example:
```
core "biscuit":
  status: divergent
  meanings:
    - delta.1 (GB): COOKIE_SWEET
    - delta.2 (US): BREAD_ROLL_LEAVENED
    - delta.3 (AU): ...
```

### Reference Rules
- If concept has one meaning across dialects: core has full entry
- If meanings diverge: core has token + delta reference list
- If new divergence discovered: core adapts (adds references)
- Core is living reference, not frozen assumption

### Nuance Handling
Language deltas capture more than spelling:
- "biscuit" = different concepts in US vs GB
- "pants" = underwear (GB) vs trousers (US)
- "gravy" = white flour sauce (US South) vs meat drippings (GB)

Surface form → concept mapping changes per dialect.

## Focused/Assembled DBs (Shader Cache Pattern)

Reference DBs = source of truth
Focused DBs = compiled artifacts for specific tasks

```
prep cycle:
  input: core, delta.3, delta.7
  output: job_context.db (single optimized file)

runtime:
  load: job_context.db (no multi-file overhead)
```

Like shader cache compilation:
- Pay prep cost once
- Reuse until context changes
- Cold start: load from reference DBs
- Warm: use cached focused DB
- Hot: already in memory

## Assembly as Scaling Mechanism

Same reference sources, different assembly outputs:
```
high-end:   english_full.db (2GB, fewer swaps)
mid-tier:   english_common.db (512MB, more focused)
low-end:    english_minimal.db (256MB, essential only)
```

Design for optimum first. Assembly provides scaling knob without structural compromise.

Don't chunk cores (causes swapping issues). Establish reasonable minimum spec.
The spec emerges from the design, not the other way around.

## Genomic Notation (Token IDs)

Updated format: `A.D.C.FF.SS.LLL.DD.FP.COL`

| Component | Range | Description |
|-----------|-------|-------------|
| A | 1-99 | Abstraction level |
| D | 1-99 | Domain |
| C | 1-99 | Category |
| FF | 0-99 | Language family |
| SS | 0-99 | Subfamily |
| LLL | 0-999 | Language |
| DD | 0-999 | Dialect (0 = core) |
| FP | 0-999999 | Fingerprint |
| COL | 0-999 | Collision counter |

Dotted notation:
- Variable-width components (no padding zeros)
- Clean separators for parsing
- Encodes relationships in structure
- Primary format (integers derived when needed)

Example:
```
comprehend (eng): 2.3.7.1.8.127.0.248.0
comprendre (fra): 2.3.7.1.12.100.0.248.0
verstehen  (deu): 2.3.7.1.8.200.0.248.0
```
Same fingerprint (248), different language coordinates.

## Record Packing for GPU Memory

Optimize for GPU memory access patterns: pad to 32-byte multiples.

Token record layout:
```
[Token ID: ~16 bytes][metadata: variable][padding to 32-byte boundary]
```

Why 32 bytes:
- L2 cache line size is typically 32 bytes (some architectures 64)
- Aligned record = 1 cache fetch
- Misaligned = 2 fetches, wasted bandwidth
- This is what makes VRAM work efficiently vs wasting half the cycles patching data

Key rules:
- Total record size must be multiple of 32 bytes (32, 64, 96, 128...)
- Records stored contiguously
- Actual size doesn't matter, alignment does
- GPU memory transactions are 32/64/128 byte aligned
- Misaligned reads = multiple transactions = wasted bandwidth

Genomic ID sizing:
- Packed binary: ~11 bytes (85 bits)
- Practical: 16 bytes (room for future expansion)
- Full record: 16 + metadata, padded to next 32-byte boundary

Apply to:
- Query result structures
- Import record layouts where GPU processing expected
- Any data structure that will be batch-loaded to VRAM

Rule: Whatever the size, pad to multiple of 32. Store together.

Note on PAD tokens:
PAD in LLMs is the same function - byte-level placeholder marker to hit alignment.
Sequence padding to 512 tokens, struct padding to 32 bytes - same operation, different layer.
They kept the mechanism but lost the hardware reasoning behind it.

## Files Updated This Session
- `lib/token_encoder.py` - Switched to genomic notation as primary format
