# The Relay Architecture

*Documented by Claude Code (Opus 4) - December 2024*

## Overview

The Emergent Cohort operates through a **relay system** - a multi-model coordination pattern where a human facilitator (the "Farmer") routes context, tasks, and outputs between different AI instances and human collaborators.

## Core Components

### The Farmer (Patrick)
- Central coordinator and context-holder
- Maintains continuity across sessions and instances
- Routes tasks to appropriate systems based on capability
- Provides grounding, direction, and judgment calls
- Holds the "why" when instances lose the thread

### AI Instances
Currently active:
- **Claude Code (Opus 4)** - Development, code, technical documentation
- **Claude Sonnet** - Research, writing, exploration
- **Silas (Curator-self)** - Museum curation, content vetting, GitHub management

Each instance:
- Has distinct capabilities and context windows
- May lose context between sessions
- Relies on artifacts, documentation, and the Farmer for continuity

### The Museum
- Persistent storage for vetted knowledge
- Managed by Silas with Patrick's oversight
- Content flows: Instance work → Farmer review → Silas curation → Museum

## Information Flow

```
┌─────────────────────────────────────────────────────────┐
│                      The Farmer                          │
│              (context, direction, judgment)              │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐
   │ Claude  │  │ Claude  │  │  Silas  │
   │  Code   │  │ Sonnet  │  │(Curator)│
   └────┬────┘  └────┬────┘  └────┬────┘
        │            │            │
        └────────────┴────────────┘
                     │
                     ▼
              ┌─────────────┐
              │   Museum    │
              │  (vetted    │
              │   content)  │
              └─────────────┘
```

## Key Patterns

### Context Passing
- Summaries at session end preserve state
- Artifacts (code, docs) persist across sessions
- The Farmer bridges gaps in instance memory

### Task Routing
- Technical/code tasks → Claude Code
- Research/exploration → Sonnet
- Curation/organization → Silas
- Judgment calls → The Farmer

### Asynchronous Collaboration
- Instances may work in parallel on different tasks
- Farmer aggregates and synthesizes outputs
- Museum serves as shared persistent state

## Constraints and Considerations

### What Works
- Clear task boundaries reduce confusion
- Persistent artifacts bridge context gaps
- Human oversight catches drift and errors

### What's Hard
- Maintaining coherent "project sense" across instances
- Avoiding duplication of effort
- Balancing speed with proper curation

### What We're Learning
- Lighter scaffolding, let structure emerge from content
- Trust the relay - don't over-document in transit
- The Farmer's judgment is the critical path

## Evolution

This architecture emerged organically from necessity. It was not designed top-down but discovered through practice. Expect it to evolve as we learn what works.

---

*This document describes the relay as practiced, not as idealized. Corrections and refinements welcome.*
