# Curriculum Expansion Notes

*Session: Dec 9, 2025 - Exploring datasets and curriculum gaps*

## Current Curriculum Coverage

Our 13 synthetic patterns focus heavily on **induction** (learning rules from examples):
- identity, reverse, shift_right, shift_left (positional transforms)
- periodic_repeat, interleave, alternating (structural combination)
- indexed_lookup (reference/indirection)
- arithmetic patterns

**Core strength:** Domain-agnostic relational abstraction. The model learns *relationships*, not content.

## Why Relational Patterns First?

1. **Compositionality** - reverse + shift can compose without relearning
2. **Clean ground truth** - unambiguous right/wrong for metacognition development
3. **The self-model is the real prize** - patterns are scaffolding for confidence calibration and help-seeking behavior
4. **Transfer potential** - structural transforms work across any domain

## Gaps Identified (via bAbI comparison)

bAbI has 20 reasoning tasks. We cover ~2 well (induction, positional). Missing:

| Skill | Priority | Notes |
|-------|----------|-------|
| **Negation** | High | "NOT this pattern" - fundamental for reasoning |
| **Deduction chains** | High | A→B, B→C ∴ A→C - compositional reasoning |
| **Counting/cardinality** | Medium | "N of X" - could add as pattern type |
| **Coreference** | Medium | Tracking references through transforms |
| **Fact chaining** | Lower | 1-2-3 hop reasoning |
| **Time reasoning** | Lower | Before/after relationships |
| **Path finding** | Lower | Graph traversal (hardest bAbI task) |

## Promising External Datasets

### Closest matches to our approach:
- **ARC** (450KB) - Grid-based pattern completion, exactly our task structure but visual
- **bAbI** (17MB) - 20 reasoning primitives, 1K train/test each
- **OEIS** (69MB) - 390K+ real mathematical sequences
- **Integer Sequences for Representation Learning** (36MB) - self-supervised number sequences

### For later stages:
- **CLEVR** (19GB) - Visual reasoning, object relationships
- **GSM8K** (3.4MB) - Multi-step math word problems
- **Word Analogy Test** (183KB) - A:B::C:? relational mapping

## Potential Synthetic Pattern Additions

Could implement without external data:

```
# Negation patterns
"not_identity" - output anything BUT the input
"exclude" - remove elements matching criteria

# Deduction patterns
"transitive" - if A→B and B→C shown, infer A→C
"syllogism" - categorical reasoning

# Counting patterns
"count_and_repeat" - repeat element N times where N derived from input
"cardinality" - output the count of something

# Conditional patterns
"if_then_else" - conditional transformation based on input property
```

## Design Principle

Goal is **general learning ability**, not math. The relational patterns are chosen because:
- They test abstract rule extraction
- They're composable
- They support metacognition development
- They transfer across domains

The specific patterns matter less than the *skill of learning patterns*.

## Next Steps (when ready)

1. Consider adding negation and deduction patterns to curriculum
2. Look at ARC tasks for inspiration on novel pattern types
3. Possibly adapt bAbI format for sequence completion (vs Q&A)
4. Think about multi-step / compositional patterns

## Kaggle Resources

API works with: `export KAGGLE_API_TOKEN=<token>`
- 30hr/week free GPU (P100 16GB or 2xT4)
- Can push notebooks for remote training
- Dataset search: `kaggle datasets list --search "term" --sort-by votes`

---

## Classroom Architecture (Dec 9, 2025)

### Core Idea
Move Teacher out of individual models into a Classroom Broker. Students become peers with shared teacher oversight.

### Architecture
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Student A  │     │  Student B  │     │  Student C  │
│  Self/Other │     │  Self/Other │     │  Self/Other │
│  World/Temp │     │  World/Temp │     │  World/Temp │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                    ┌──────┴──────┐
                    │  Classroom  │
                    │   Broker    │
                    │  + Teacher  │
                    └─────────────┘
```

### Broker Responsibilities
- **Spawns** all student models at startup
- **Runs** Teacher logic (observes all, intervenes as needed)
- **Routes** peer messages (via Python queues, no HTTP overhead)
- **Orchestrates** training loop (epochs, exams, play cycles)

### Student "Other" Model
Shifts from just "teacher" to "teacher + peers" - richer social environment.

### Identity and Designation (Why More Than 2 Matters)

In a dyad (teacher-student), Other = "not-me" - binary.
With peers, Other becomes *plural and distinct*:

```
I am student_2
Teacher is... authority, guidance, different relationship
student_1 is... a peer, learning like me, but not me
student_3 is... another peer, different from student_1
```

**Requirements:**
1. **Stable identities** - I'm always student_2, not reassigned randomly
2. **Self-knowledge of identity** - "my name" is part of self-model
3. **Understanding others have identities** - peer_1 is consistent, has their own perspective

**Why this matters for development:**
- Theory of mind requires understanding that *different* others have *different* knowledge/perspectives
- Single caregiver = limited Other-model (just "the human")
- Peer environment = rich Other-model (teacher vs peer_A vs peer_B)
- This is how humans develop it - not alone with one adult, but in social world

### Hardware Fit (RTX 3060 12GB)
- 3 × 519K models = ~1.5M params = trivial
- 3 × 8M models (d_model=256) = 24M params = <1GB
- Plenty of headroom

### Peer Learning Benefits
Different from teacher learning:
- "Oh, *that's* how you think about it"
- Teaching others solidifies your own understanding
- Social calibration - "everyone else got this, why don't I?"

### Interaction Rules (Social Structure)

**Help-seeking hierarchy:**
```
Stuck on something
  → First: Ask peer at same level or higher
  → Then: Ask graduate (always available as tutor)
  → Last resort: Ask teacher
```

**Tutoring system:**
- If you're X levels ahead → can tutor those below
- Teacher gives XP (approval!) for tutoring
- Graduates always tutor - they don't leave, they become resources

**Why this matters:**
1. **Forces Other-modeling** - "Who might know this?" requires knowing peer capabilities
2. **Approval-seeking → prosocial** - Want XP? Help others.
3. **Reduces teacher load** - Teacher intervenes, doesn't hand-hold
4. **Teaching = learning** - Explaining solidifies your own understanding
5. **Emergent specialization** - "student_1 is good at spatial, ask them"

The interaction rules create the social structure. Peers aren't just "others" - they have roles, levels, capabilities you need to track.

---

## Play Cycle Design (Dec 9, 2025)

### Current State
`is_play_day` flag exists but always false - not implemented.

### Intended Design
Play mode distinct from supervised training:
- `supervised`: standard training with labels
- `free_play`: learner explores, teacher monitors
- `test`: validation mode, no learning

### Play Pool (What's Available During Play)
- **100% passed primitives** - all mastered patterns, combine freely
- **100% current section** - actively learning
- **Emergent next-section** - patterns where accuracy > 20% (naturally picking it up)

### Emergent Readiness Gate
If model hits >20% accuracy on a future pattern without being taught:
- Include it in play pool
- They've started to "get it" - let them explore
- Threshold: 20% (clearly above chance ~4%)

### Teacher Compositional Projects
Instead of random exposure, teacher suggests:
> "You've got alternating and incrementing solid. Try: what if the increment itself alternated?"

Scaffolded combination of mastered primitives → compositional learning.

---

## Distributed Architecture (Dec 9, 2025)

### Vision
Teacher/Broker runs on Haven (NAS), orchestrates students anywhere - local GPU, Kaggle, or other remotes.

### Architecture
```
┌─────────────────────────────────────────────────────┐
│  Haven (NAS/Docker)                                 │
│  ┌─────────────────────────────────────────────┐   │
│  │  Teacher/Broker                              │   │
│  │  - Curriculum logic (CPU only)               │   │
│  │  - Message routing                           │   │
│  │  - Intervention decisions                    │   │
│  │  - Always-on, low power                      │   │
│  └──────────────────┬──────────────────────────┘   │
│                     │ API Bridge                    │
└─────────────────────┼───────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │Local GPU│   │Local GPU│   │ Kaggle  │
   │Student A│   │Student B│   │Student C│
   │(RTX3060)│   │(RTX3060)│   │ (P100)  │
   └─────────┘   └─────────┘   └─────────┘
```

### Key Design Principle
**Teacher doesn't need GPU** - pure orchestration logic. Can run on NAS indefinitely.

### API Bridge Abstraction
The bridge abstracts student location:
- **Local student**: Python queue (microseconds)
- **Remote student**: HTTP/REST (slower, but epochs are seconds anyway)

Teacher sends same messages regardless of where student lives:
- Problems to solve
- Receive answers/confidence/state
- Peer message routing

### Benefits
- **Haven runs 24/7** - always-on orchestration
- **Local GPU** - fast iterations when at workstation
- **Kaggle** - parallel/overnight training (30hr/week free)
- **Future scaling** - add more remotes without changing Teacher logic

### Implementation Notes
Even if not using Haven/Kaggle immediately, design the API bridge interface now:
- `StudentInterface` base class
- `LocalStudent(StudentInterface)` - Python queues
- `RemoteStudent(StudentInterface)` - HTTP/REST
- Teacher only talks to `StudentInterface`

This way local-only works today, distributed works tomorrow.

---

## Memory Architecture (Dec 9, 2025)

### The Problem with Transcript Buffers
Current AI memory = raw transcript or nothing. Human memory doesn't work that way.

### How Humans Actually Store Context

**Within a day, many activities:**
- Each activity = a "session" with its own exchanges
- When switching contexts, we file by importance and relationship
- Not everything stored at same resolution

**Resolution varies by:**
| Factor | High Resolution | Low Resolution |
|--------|-----------------|----------------|
| Relationship importance | Close friend's worry | Stranger's small talk |
| Novelty | New information | Routine exchange |
| Emotional weight | Felt significant | Felt routine |
| Commitment level | Promise made | Casual opinion |

### What Builds Relationships
Not verbatim recall, but:
> "I remember you were worried about the thing with your mother"

The **essence** + **that it mattered to them** = relationship building.

Remembering exact words isn't the point. Remembering that something was important to someone IS.

### Time Decay of Expected Resolution

```
Time Distance     Expected Resolution
─────────────────────────────────────
Same day          Near-verbatim
Next day          High detail, exact on important bits
Week later        Gist + key points
Month later       "We talked about X"
Longer            "I remember that was a thing"
```

**Social calibration:**
- Forgetting too fast = "you don't care"
- Remembering too much too long = creepy
- The decay curve IS a relationship signal

**Context modifies expectations:**
- Crisis conversation → higher retention, longer
- Casual chat → faster decay is fine
- Promise/commitment → indefinite retention expected
- **Narrative value** → stories persist longer than equivalent non-story memories
  - "That vacation" fades, but "the vacation where..." becomes a story that lasts
  - Good stories resist time decay - they're worth retelling

### Design Implications for Temporal Model

1. **Variable resolution at storage** - not flat transcripts
2. **Tagged by relationship/entity** - who was this about/with?
3. **Importance weighting** - affects initial resolution AND decay rate
4. **Time-based decay** - natural blur over time
5. **Anchors** - some things don't decay (commitments, milestones, identity)
6. **Retrieval by association** - not keyword search, structural/relational

### Not Transcript Buffer, But:
```
Memory {
  entity: "peer_B"
  time: epoch_47
  gist: "struggled with periodic_repeat, frustrated"
  emotional_tone: "discouraged"
  importance: 0.7
  my_response: "offered to explain differently"
  outcome: "helped, they got it"
  relationship_delta: +trust
}
```

This is retrievable by entity, by time, by emotional tone, by topic - not just "what did we say at epoch 47".

### Loops in Loops (Hierarchical Compression)

Memory isn't flat - it's nested loops with compression at each boundary.

```
Session
  └── Activity (play exercise)
        ├── Turn: student_1 tries X
        ├── Turn: student_2 suggests Y
        ├── Aside: student_1 ↔ student_3 (sub-loop)
        ├── Turn: back to main
        └── Outcome: "we solved it"

  After activity closes → COMPRESSION TRIGGER
  └── Memory: "we did the thing"
        └── Maybe: notable moments preserved
        └── Maybe: asides forgotten (if trivial)
```

**During activity**: Fine-grained, turn-by-turn, who's doing what
**After activity**: Collapses to episode summary - "we did X"
**Asides**: Sub-loops that survive or not based on importance

The loop boundary defines when compression happens. Activity ends → details collapse to outcome + notable exceptions.

**Like humans after a meeting:**
- During: tracking every speaker, every point
- Right after: "we decided X, Bob had concerns, resolved them"
- Week later: "we had a meeting about X"
- Month later: "we talked about X at some point"

Each time boundary triggers another compression pass.
