# Physics-Aware AI Models Research Report
## Generating Physical Scenarios and Simulations with Neural Networks

**Date:** December 2024
**Objective:** Identify HuggingFace models and datasets suitable for generating physically valid scenarios (pendulum motion, projectile trajectories, bouncing dynamics, etc.)

---

## Executive Summary

This research identifies multiple promising approaches for generating physics scenarios using neural networks. The landscape includes:

1. **World Models** - Generative models trained on physics interactions that understand causality
2. **Physics-Grounded Video Generation** - Diffusion models conditioned on physics parameters
3. **Physics-Informed Neural Networks (PINNs)** - Neural operators that encode physical laws
4. **Graph Neural Networks** - Particle-based simulators using message passing
5. **Large-Scale Physics Datasets** - 15TB+ collections for training custom models

**Bottom Line:** The most promising paths forward are:
- Fine-tune video diffusion models (PhysCtrl approach) conditioned on physics parameters
- Use Graph Neural Networks (GNS) for particle-based scenarios
- Leverage The Well dataset for training custom neural operators
- Combine LLMs with physics reasoning for scenario description generation

---

## RTX 3060 (12GB) Feasibility Summary

### Will Run (Green Light)
| Model | Size | Notes |
|-------|------|-------|
| **ReVision** | 1.5B | Best choice, runs with room to spare |
| **GNS** | Variable | Graph networks are memory efficient |
| **FNO variants** | Small | Many implementations fit in 12GB |
| **Llama-2-7B-physics** | 7B | Runs with quantization |
| **Small PINNs** | <1B | Light-weight neural operators |

### Possible with Optimization (Yellow Light)
| Model | Size | Notes |
|-------|------|-------|
| **PhysCtrl** | ~10-13B | Needs attention slicing, FP16 |
| **PhysGen** | ~10-13B | Similar to PhysCtrl |
| **Llama-2-13B-physics** | 13B | Only with 4-bit quantization |

### Won't Run (Red Light)
| Model | Size | Notes |
|-------|------|-------|
| **WoW** | 14B | Exceeds capacity even quantized |
| **Large world models** | 20B+ | Too large |

---

## Top Recommendations for Physics Scenario Generation

### 1. Graph Neural Network Simulator (GNS) - RECOMMENDED

**Source:** DeepMind
**Paper:** https://arxiv.org/abs/2211.10228

**What it does:**
- Learns to simulate fluids, rigid solids, deformable materials
- Represents state as particle nodes, interactions as edges
- Message passing computes dynamics

**Why it's best for your use case:**
- **5,000x faster** than traditional simulators
- Generalizes to unseen initial conditions
- Scales to thousands of particles
- Memory efficient on RTX 3060
- Perfect for bouncing, pendulum, projectile scenarios

### 2. ReVision (1.5B) - Best Video Model for RTX 3060

**Paper:** https://huggingface.co/papers/2504.21855

**What it does:**
- 3D-aware video diffusion with physics priors
- Only 1.5B parameters - outperforms 13B+ models

**Why it works:**
- Smallest viable physics-aware video model
- Direct HuggingFace integration
- Good starting point for prototyping

### 3. PhysicsNeMo Framework

**Repository:** https://github.com/NVIDIA/physicsnemo

**What it provides:**
- Neural operators, GNNs, Transformers, PINNs
- PyTorch-based, optimized for NVIDIA GPUs
- Apache 2.0 license

**Why it works:**
- Purpose-built for physics simulation
- Can run on RTX 3060
- Active NVIDIA support

---

## Key Datasets

### The Well - 15TB Physics Collection
**HuggingFace:** https://huggingface.co/collections/polymathic-ai/the-well-67e129f4ca23e0447395d74c

- 16 datasets covering fluid dynamics, materials, acoustics, etc.
- Unified format, PyTorch interface
- Stream from hub (no local storage needed)
- Perfect for training custom models

### camel-ai/physics - 20K Problem-Solution Pairs
**HuggingFace:** https://huggingface.co/datasets/camel-ai/physics

- 25 physics topics, GPT-4 generated solutions
- Use for training LLMs to describe scenarios

### Physics-Tuned LLMs

| Model | Base | Use Case |
|-------|------|----------|
| Llama-2-7B-physics | Llama-2-7b | Generate scenario descriptions |
| PhysicsLlama-13B | Llama-2-13b | Complex physics reasoning |

---

## Recommended Architecture for Your System

```
┌─────────────────────────────────────────────────────────────┐
│           LLM (Llama-2-7B-physics)                          │
│           "Generate pendulum with 1m length, 45° release"   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ (physics parameters)
┌─────────────────────────────────────────────────────────────┐
│           GNS or Physics Engine                             │
│           Simulates particle dynamics                       │
│           Runs on NAS                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ (trajectories)
┌─────────────────────────────────────────────────────────────┐
│           Physics Playground (existing)                     │
│           Quantizes to training sequences                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           Student Model                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Gaps Identified

1. **No off-the-shelf physics scenario generator** - requires custom training/fine-tuning
2. **Most physics models trained on complex domains** - basic mechanics (pendulum, ball) underrepresented
3. **Recent papers (2024-2025) not yet released** - PhysCtrl, PhysGen weights pending
4. **Validation challenge** - hard to verify physical correctness automatically

---

## Next Steps

### Immediate
1. Benchmark GNS implementation on RTX 3060
2. Set up PhysicsNeMo environment
3. Stream subset of The Well dataset

### Short-term
1. Generate synthetic training data (PyBullet/Newton)
2. Fine-tune ReVision with physics parameters
3. Build validation framework

### Medium-term
1. Integrated pipeline: LLM → Physics params → GNS → Training sequences
2. Optimize for real-time inference

---

## Key Links

**Models:**
- [ReVision](https://huggingface.co/papers/2504.21855) - 1.5B physics video
- [PhysCtrl](https://huggingface.co/papers/2509.20358) - Physics-grounded generation
- [Llama-2-7B-physics](https://huggingface.co/Harshvir/Llama-2-7B-physics)

**Datasets:**
- [The Well](https://huggingface.co/collections/polymathic-ai/the-well-67e129f4ca23e0447395d74c)
- [camel-ai/physics](https://huggingface.co/datasets/camel-ai/physics)

**Frameworks:**
- [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo)
- [GNS](https://www.deepmind.com/open-source/learning-to-simulate-complex-physics-with-graph-networks)
- [Newton Engine](https://github.com/newton-physics/newton)

---

*Research compiled by explorer agent, December 10, 2025*
