# Textbook Design Principles

*Research compiled December 2024*

## Information Architecture

- **Hierarchical organization**: Chapters → Sections → Subsections
- **Consistent design patterns**: All chapters follow same template
- **Pedagogical scaffolding**: Simple → Complex, building on prior knowledge
- **Clear navigation**: Labels, visual hierarchy, predictable structure

## Worked Examples: Placement and Ratio

### The Worked Example Effect
Students learn more effectively studying worked examples BEFORE attempting similar problems.

### Example-Problem Pairing
- One complete worked example → Similar problem to solve
- **Ratio varies by expertise**:
  - Novices: More examples (high example ratio)
  - Intermediate: Mix of examples and problems
  - Advanced: Fewer examples, more independent work

### Fading Strategy
1. Complete worked examples (all steps shown)
2. Partially completed examples (students fill in missing steps)
3. Full problems (no example provided)

## Problem Set Design

### Progression Within Sets
- **Easy-to-hard ordering**: Basic application → Complex multi-step
- **Single-task to multi-task**: One skill → Combination of skills
- **Concrete to abstract**: Concrete examples → Abstract scenarios

### Quality Over Quantity
- Problems should require **ingenuity to set up**, not just calculation
- Avoid superfluous difficulty
- Test **conceptual understanding**, not rote mechanics

## Visual Design Elements

### Callout Boxes
- Draw attention to key points
- Use hierarchy: Critical (rare) → Warning → Note → Tip
- Minimize visual styling - avoid bubbles, excessive backgrounds

### Margin Notes
- Supplementary context without interrupting main text
- Definitions, clarifications, connections
- Smaller font, sans-serif for legibility

### Diagrams
- Pair directly with relevant text
- Focus on concept illustration, not decoration
- In math: visual before symbolic

## Scaffolding: Building Complexity

### Zone of Proximal Development
Gap between what students can do independently vs with support.
Effective textbooks work within this zone.

### Strategies
1. Break complex ideas into manageable parts
2. Use modeling and explicit instruction
3. Graphic organizers
4. Exemplars and worked examples
5. Task breakdown into smaller parts

### The Fading Process
- Scaffolding = temporary support structure
- Responsibility shifts from instructor → student
- Fading must match learner progression

## The Textbook Problem: Common Failures

1. **Superfluous illustrations** - decorative, not educational
2. **Lack of retrieval practice** - no embedded quizzing
3. **Blocked structure only** - topics never revisited before exam
4. **Missing metacognitive guidance** - don't teach HOW to learn
5. **Poor problem sets** - tedious calculation over insight
6. **Content sequencing** - convention over pedagogy

## Singapore Math: Why It Works

### Concrete-Pictorial-Abstract (CPA) Approach
Based on Jerome Bruner's learning theory:

1. **Concrete**: Hands-on with physical objects
2. **Pictorial**: Drawing visual representations
3. **Abstract**: Using numbers and symbols

### Key Design Features

**Fewer Topics, Greater Depth**
- Master before moving forward
- No redundant re-teaching
- Contrast: US spiral vs Singapore mastery

**Visual and Accessible**
- Short textbooks (80-190 pages)
- Lots of diagrams explaining concepts
- Clean, uncluttered aesthetic
- High visual-to-text ratio

**The Bar Model Method**
- Visual tool for representing mathematical relationships
- Identifies knowns and unknowns
- Bridges pictorial → abstract phases

**Intentional Sequencing**
- Topics revisited over years
- Each iteration builds complexity
- Students see connections across levels

### Results
- Singapore students rank #1 in TIMSS internationally
- U.S. studies show "substantial gains" across all student groups

## Interleaving vs Blocking

### Definitions
- **Blocking**: Group problems by type (all prisms, then all cylinders)
- **Interleaving**: Mix problems from different categories

### Research Findings

**Long-term retention (KEY FINDING)**:
- Day of learning: Blocked = higher accuracy
- 24 hours later: **Interleaved = 77% vs Blocked = 38%**
- One week later: Interleaved advantage even larger

**Why the reversal?** Blocked creates illusion of mastery that doesn't transfer or retain.

### When Blocking Works Better
- Rule-based learning (finding commonalities)
- Explicit categorization tasks
- Initial learning (but advantage disappears quickly)

### Why Interleaving Works
1. Forces retrieval from long-term memory
2. Prevents fluency illusion
3. Deeper processing (which approach for each problem?)
4. Better metacognitive monitoring

### Implementation
- Mix related problems from previous chapters with current
- Problem sets within chapters should mix topics
- End-of-chapter includes prior chapter material

## Application to AI Curriculum

### Current Approach (Coherence Lab)
- Abstract only (no concrete/pictorial phase)
- Phased sections = mastery before advancement ✓
- Playdays = retrieval practice ✓
- Problem sets are blocked by pattern type

### Potential Improvements
1. **Interleave during training**: Mix pattern types in batches
2. **Visual representations**: Could bar models help transformers?
3. **Example-problem pairs in playdays**: Show → Predict → Confirm
4. **Fewer patterns, greater depth**: Already doing this via phased sections

### The CPA Question
For neural networks, is there an equivalent to:
- **Concrete**: Physical manipulation (position encoding?)
- **Pictorial**: Visual representation (attention maps?)
- **Abstract**: Symbolic (current approach)

Or does the abstract-only approach work because neural nets don't have human cognitive architecture?

## Key Takeaways

1. **Worked examples before problems** - reduces cognitive load
2. **Interleave for retention** - blocked feels better, interleaved works better
3. **Mastery before advancement** - Singapore approach validated
4. **Fewer topics, greater depth** - resist the urge to cover everything
5. **Visual clarity over decoration** - every element should serve learning
6. **Scaffold and fade** - support that gradually disappears
