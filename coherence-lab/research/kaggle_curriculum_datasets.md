# Kaggle Curriculum Datasets: Multi-Subject Learning

*Research compiled December 2024*

## The Vision

Instead of inventing custom pattern encodings, use real human learning materials:
- **Each Kaggle notebook = a subject**
- **School day = rotating through subjects** (interleaved learning)
- **CPA approach**: Concrete experience before abstract symbols

---

## Physics Playground (Recess/Embodied Learning)

The "Concrete" in CPA. Kids learn physics through swings, throwing balls, slides, block towers.

### Recommended Datasets

| Dataset | Link | Why It Works |
|---------|------|--------------|
| **MotionSense** | kaggle.com/datasets/malekzadeh/motionsense-dataset | Smartphone accelerometer data from walking, running, stairs. *Embodied* physics - what movement feels like |
| **Damped Harmonic Oscillator** | kaggle.com/datasets/charel/damped-harmonic-oscillator-time-series | Swing physics! Position, velocity, acceleration over time |
| **Simple Pendulum** | kaggle.com/datasets/usharengaraju/simple-pendulum-time-vs-period | Length affects swing period - playground discovery |
| **Projectile Motion** | kaggle.com/datasets/shashwatwork/projectile-motion-equation-dataset | Throwing things - trajectories, gravity |
| **Physics Attractor** | Search: physics attractor time series | Chaotic but patterned systems |

### Curriculum Ideas
- Start with MotionSense (embodied, relatable)
- Move to pendulum (systematic observation)
- Then projectile (prediction based on initial conditions)

---

## Music Class

Rhythm, pattern, repetition - deeply mathematical but experiential first.

### Recommended Datasets

| Dataset | Link | Why It Works |
|---------|------|--------------|
| **Music Notes Dataset** | kaggle.com/datasets/karnikakapoor/music-notes | Basic rhythm and note concepts - **beginner friendly** |
| **MAESTRO** | kaggle.com/datasets/jackvial/themaestrodatasetv2 | 200 hours piano, MIDI + audio aligned. Process AND product |
| **MusicNet** | kaggle.com/datasets/imsparsh/musicnet-dataset | Note sequences with beat structure |
| **Lakh MIDI** | kaggle.com/datasets/iamjkp/lakh-midi-dataset-17 | 176k MIDI files for pattern extraction |
| **GTZAN** | kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification | Audio features, MFCCs - more abstract |

### Curriculum Ideas
- Start with Music Notes (basic patterns)
- Rhythm before melody
- Simple sequences before complex pieces

---

## Art Class

Visual pattern, shape recognition, creativity.

### Recommended Datasets

| Dataset | Link | Why It Works |
|---------|------|--------------|
| **Four Shapes** | kaggle.com/datasets/smeschke/four-shapes | Circles, squares, triangles, stars - **kindergarten basics** |
| **Quick Draw Doodle** | kaggle.com/competitions/quickdraw-doodle-recognition | 50M drawings as *stroke vectors* - shows process not just result |
| **Doodle Dataset** | kaggle.com/datasets/ashishjangra27/doodle-dataset | 340 categories, 1M images |
| **Hand-drawn Shapes** | kaggle.com/datasets/imbikramsaha/hand-drawn-shape-classification | Shape recognition from messy input |
| **Geometric Shapes** | kaggle.com/datasets/cactus3/geometric-shapes-dataset | Clean geometric forms |

### Curriculum Ideas
- Start with Four Shapes (basic vocabulary)
- Quick Draw for "how to draw" (process)
- Geometric shapes for precision

---

## Reading Class (Future Research)

Potential datasets to explore:
- Children's book text corpora
- Graded reading level datasets
- Phonics and word pattern datasets

---

## Integration: The School Day

```
Morning:
  - Math (Singapore Math patterns)
  - Reading (word patterns)

Recess:
  - Physics Playground (MotionSense, pendulum)

Afternoon:
  - Music (rhythm patterns)
  - Art (shape patterns)
```

### Interleaving Strategy
- Don't finish one subject before starting another
- Rotate through subjects within each training session
- Let patterns from one subject reinforce another
  - Math: sequences, counting
  - Music: rhythm sequences, counting beats
  - Physics: position sequences, counting oscillations
  - Art: shape sequences, counting vertices

---

## Next Steps

1. [ ] Create Kaggle notebook for Physics (MotionSense + Pendulum)
2. [ ] Create Kaggle notebook for Music (Music Notes + simple MIDI)
3. [ ] Create Kaggle notebook for Art (Four Shapes + Quick Draw)
4. [ ] Design interleaved training loop that rotates subjects
5. [ ] Align difficulty across subjects (kindergarten → first grade → ...)

---

## Key Insight

> "Why are we reinventing anything? Why not just use exactly what humans use?"

The curriculum shouldn't be designed from scratch. It should be **curated** from real human learning materials. Kaggle gives us access to the data; our job is to sequence it developmentally.
