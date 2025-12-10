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

### Top Notebooks (Evaluated)

| Notebook | URL | Why It Works |
|----------|-----|--------------|
| **Run or Walk** | [kaggle.com/code/vmalyi/run-or-walk-data-analysis-and-visualization](https://kaggle.com/code/vmalyi/run-or-walk-data-analysis-and-visualization) | **TOP PICK** - Kids relate to running/walking. Clear accelerometer patterns. Excellent visualizations. |
| **Human Activity Recognition** | [kaggle.com/code/malekzadeh/human-activity-recognition-with-mobile-sensing](https://kaggle.com/code/malekzadeh/human-activity-recognition-with-mobile-sensing) | 6 activities (stairs, walking, jogging, sitting). Embodies physics. Research-backed. |
| **EDA Human Activity** | [kaggle.com/code/abheeshthmishra/eda-of-human-activity-recognition](https://kaggle.com/code/abheeshthmishra/eda-of-human-activity-recognition) | Great visualization focus. 561 features from beginner→advanced. |
| **Simple Pendulum Simulator** | [kaggle.com/code/bensalem14/simple-pendulum-simulator](https://kaggle.com/code/bensalem14/simple-pendulum-simulator) | Pure physics simulation. Oscillation/swing intuition. Bridge to math. |
| **Projectile Motion** | [kaggle.com/code/yaserrahmati/projectile-motion](https://kaggle.com/code/yaserrahmati/projectile-motion) | Throwing things! Parabolic trajectories. Observable physics. |

### Recommended Datasets

| Dataset | Link | Why It Works |
|---------|------|--------------|
| **Run or Walk** | [kaggle.com/datasets/vmalyi/run-or-walk](https://kaggle.com/datasets/vmalyi/run-or-walk) | 50Hz accelerometer, 2-sec windows. Featured by Kaggle. |
| **MotionSense** | [kaggle.com/datasets/malekzadeh/motionsense-dataset](https://kaggle.com/datasets/malekzadeh/motionsense-dataset) | 24 participants, 6 activities. Gravity separated from user acceleration. |
| **UCI Smartphones** | [kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones](https://kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones) | 30 participants, 561 features. Standard benchmark. |

### Curriculum Path
1. **Run or Walk** - Most concrete/relatable (every child runs and walks)
2. **MotionSense** - Structured learning (6 clear activities)
3. **Simple Pendulum** - Physics simulation (oscillation concepts)

---

## Music Class

Rhythm, pattern, repetition - deeply mathematical but experiential first.

### Top Notebooks (Evaluated)

| Notebook | URL | Why It Works |
|----------|-----|--------------|
| **Beginner's Guide to Audio Data** | [kaggle.com/code/fizzbuzz/beginner-s-guide-to-audio-data](https://kaggle.com/code/fizzbuzz/beginner-s-guide-to-audio-data) | **TOP PICK** - SEE and HEAR patterns. Audio visualization. 2018 classic, frequently referenced. |
| **Music Generation for Beginners** | [kaggle.com/code/naklecha/music-generation-guide-for-beginners](https://kaggle.com/code/naklecha/music-generation-guide-for-beginners) | Classical MIDI. Teaches generation from sequences. |
| **Play Audio, Create Spectrogram** | [kaggle.com/code/vbookshelf/play-audio-read-the-files-create-a-spectrogram](https://kaggle.com/code/vbookshelf/play-audio-read-the-files-create-a-spectrogram) | Three modes: play sound, read data, visualize. Essential for embodied learning. |
| **Tempo & Beat Tracking** | [kaggle.com/code/enrcdamn/tempo-estimation-and-beat-tracking-pipeline](https://kaggle.com/code/enrcdamn/tempo-estimation-and-beat-tracking-pipeline) | Core rhythm extraction. See beats being identified. |
| **MAESTRO metadata/midi/wav** | [kaggle.com/code/robbynevels/maestro-metadata-wav-midi-performance-events](https://kaggle.com/code/robbynevels/maestro-metadata-wav-midi-performance-events) | 200 hours piano. Perfect MIDI-audio alignment. Rhythm/performance connection. |
| **Noobiano: Intro to Music AI** | [kaggle.com/code/aleksandrsigalov/noobiano-beginner-introduction-to-music-ai](https://kaggle.com/code/aleksandrsigalov/noobiano-beginner-introduction-to-music-ai) | Explicitly beginner-focused. Good first notebook. |

### Recommended Datasets

| Dataset | Link | Why It Works |
|---------|------|--------------|
| **Music Notes** | [kaggle.com/datasets/kishanj/music-notes-datasets](https://kaggle.com/datasets/kishanj/music-notes-datasets) | Note values (half, full, etc). **Beginner friendly**. |
| **MAESTRO v2** | [kaggle.com/datasets/jackvial/themaestrodatasetv2](https://kaggle.com/datasets/jackvial/themaestrodatasetv2) | 200 hours piano, MIDI + audio aligned. Process AND product. |
| **Classical Music MIDI** | [kaggle.com/datasets/soumikrakshit/classical-music-midi](https://kaggle.com/datasets/soumikrakshit/classical-music-midi) | Clean MIDI for pattern learning. |

### Curriculum Path
1. **Beginner's Guide to Audio** - SEE and HEAR patterns simultaneously
2. **Tempo & Beat Tracking** - Learn rhythm detection in real music
3. **MAESTRO MIDI** - Understand note sequences and melodies
4. **Music Generation** - Create new patterns from learned concepts

---

## Art Class

Visual pattern, shape recognition, creativity. **Quick Draw is special** - it captures HOW things are drawn (stroke sequences), not just final pictures.

### Top Notebooks (Evaluated)

| Notebook | URL | Why It Works |
|----------|-----|--------------|
| **Getting Started: Quick Draw** | [kaggle.com/code/inversion/getting-started-viewing-quick-draw-doodles-etc](https://kaggle.com/code/inversion/getting-started-viewing-quick-draw-doodles-etc) | **TOP PICK** - Official intro. Stroke-by-stroke visualization. See HOW people draw. |
| **Let's Play With Quick Draw!!** | [kaggle.com/code/harunshimanto/lets-play-with-quick-draw](https://kaggle.com/code/harunshimanto/lets-play-with-quick-draw) | **Most engaging** - Animated strokes. Watch drawings happen in real-time. |
| **Raw Stroke Sequences 1D-CNN** | [kaggle.com/morrisb/raw-stroke-sequences-in-1d-cnn/data](https://kaggle.com/morrisb/raw-stroke-sequences-in-1d-cnn/data) | Treats strokes as SEQUENCES not images. Temporal patterns in art. |
| **Four Shapes** | [kaggle.com/code/oysiyl/four-shapes](https://kaggle.com/code/oysiyl/four-shapes) | Simplest geometry. Circle, square, triangle, star. Absolute beginner. |
| **Hand Drawn Shapes CNN** | [kaggle.com/code/maherdissem/hand-drawn-shapes-classification-with-cnn](https://kaggle.com/code/maherdissem/hand-drawn-shapes-classification-with-cnn) | Imperfect human drawings still work. Teaches visual abstraction. |
| **HDS Shapes ETL** | [kaggle.com/code/frobert/hds-shapes-etl-and-classify](https://kaggle.com/code/frobert/hds-shapes-etl-and-classify) | Triangle, rectangle, ellipse. Hand-drawn geometric properties. |

### Recommended Datasets

| Dataset | Link | Why It Works |
|---------|------|--------------|
| **Quick Draw Doodles** | [kaggle.com/c/quickdraw-doodle-recognition](https://kaggle.com/c/quickdraw-doodle-recognition) | 50M drawings as stroke vectors. **Shows process, not just result.** |
| **Four Shapes** | [kaggle.com/datasets/smeschke/four-shapes](https://kaggle.com/datasets/smeschke/four-shapes) | Circles, squares, triangles, stars. **Kindergarten basics.** |
| **Hand-drawn Shapes** | [kaggle.com/datasets/imbikramsaha/hand-drawn-shape-classification](https://kaggle.com/datasets/imbikramsaha/hand-drawn-shape-classification) | Shape recognition from messy input. |

### Curriculum Path
1. **Four Shapes** - Basic geometric vocabulary
2. **Hand Drawn Shapes** - Human variation still recognizable
3. **Let's Play With Quick Draw** - See drawing as animation/process
4. **Getting Started Quick Draw** - Explore stroke sequences in depth
5. **Raw Stroke Sequences** - Analyze drawing order as data

### Key Insight: Why Quick Draw Is Special

Google recorded **billions of actual drawing sessions**. Unlike image datasets (final results only), Quick Draw shows:
- How different people draw the same object differently
- The temporal/sequential nature of drawing (which stroke comes first?)
- Why neural networks on strokes sometimes beat those on images

This is teaching the *process* of visual creation.

---

## Reading Class

Letter recognition → phonics → words → sentences → comprehension. The foundation of everything.

### Top Datasets (Evaluated)

| Dataset | URL | Why It Works |
|---------|-----|--------------|
| **A-Z Handwritten Alphabets** | [kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format](https://kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format) | 370K+ letters. Foundation for letter recognition. K-1. |
| **Alphabet Characters Fonts** | [kaggle.com/datasets/thomasqazwsxedc/alphabet-characters-fonts-dataset](https://kaggle.com/datasets/thomasqazwsxedc/alphabet-characters-fonts-dataset) | 390K letters in multiple fonts. Letter constancy. |
| **Phoneme Dataset** | [kaggle.com/datasets/timrie/phoneme](https://kaggle.com/datasets/timrie/phoneme) | Audio phoneme recordings. Letter sounds. Phonics. |
| **Children Stories Text Corpus** | [kaggle.com/datasets/edenbd/children-stories-text-corpus](https://kaggle.com/datasets/edenbd/children-stories-text-corpus) | **TOP PICK** - Real children's books. Word families, rhymes. Grade 1-3. |
| **CommonLit Readability** | [kaggle.com/c/commonlitreadabilityprize](https://kaggle.com/c/commonlitreadabilityprize) | 5000+ passages with Lexile scores. **Graded difficulty**. |
| **Highly Rated Children Books** | [kaggle.com/datasets/thomaskonstantin/highly-rated-children-books-and-stories](https://kaggle.com/datasets/thomaskonstantin/highly-rated-children-books-and-stories) | Age-tagged books. K-5 progression. |
| **15000 Gutenberg Books** | [kaggle.com/mateibejan/15000-gutenberg-books](https://kaggle.com/mateibejan/15000-gutenberg-books) | Largest corpus. Extract word patterns, rhymes, frequency. |

### Curriculum Path
1. **Letter Recognition** (K): A-Z Handwritten + Alphabet Fonts
2. **Phonics** (K-1): Phoneme Dataset - letter sounds
3. **Words** (1-2): Children Stories - word families ("cat, bat, hat")
4. **Sentences** (2-3): CommonLit Readability - graded passages
5. **Comprehension** (3+): Children's Book Test - context-based questions

---

## Writing & Spelling Class

Letter formation → spelling patterns → word building → writing.

### Top Datasets (Evaluated)

| Dataset | URL | Why It Works |
|---------|-----|--------------|
| **EMNIST Letters** | [kaggle.com/datasets/crawford/emnist](https://kaggle.com/datasets/crawford/emnist) | **TOP PICK** - Professional letter/digit recognition. 103K+ characters. |
| **English Word Frequency** | [kaggle.com/datasets/rtatman/english-word-frequency](https://kaggle.com/datasets/rtatman/english-word-frequency) | 333K words by frequency. Filter for sight words. |
| **Words and Syllables** | [kaggle.com/datasets/arnavsharmaas/words-and-their-syllables](https://kaggle.com/datasets/arnavsharmaas/words-and-their-syllables) | 2900+ words with syllable counts. Phonetic patterns. |
| **Student Writing Errors** | [kaggle.com/datasets/ziya07/student-writing-error-correction-dataset](https://kaggle.com/datasets/ziya07/student-writing-error-correction-dataset) | Real errors with corrections. Grammar + spelling rules. |
| **Spelling Corrector** | [kaggle.com/datasets/bittlingmayer/spelling](https://kaggle.com/datasets/bittlingmayer/spelling) | Error → correction pairs. Spelling patterns. |
| **Dyslexia Handwriting** | [kaggle.com/datasets/drizasazanitaisa/dyslexia-handwriting-dataset](https://kaggle.com/datasets/drizasazanitaisa/dyslexia-handwriting-dataset) | Real handwriting variation. Remediation support. |

### Curriculum Path
1. **Letter Formation** (K): EMNIST + A-Z Handwritten
2. **High-Frequency Words** (K-1): Word Frequency lists (Dolch, Fry)
3. **Syllable Patterns** (1-2): Words and Syllables
4. **Spelling Rules** (2-3): Spelling Corrector - error patterns
5. **Writing Conventions** (3+): Student Writing Errors - grammar

---

## OER Curriculum Resources (Scope & Sequence)

Real K-12 curricula for understanding WHAT to teach WHEN.

### Primary Resources

| Resource | URL | Why It Works |
|----------|-----|--------------|
| **Eureka Math / EngageNY** | [embarc.online](https://embarc.online/) | **BEST for sequencing** - Explicit module structure K-5. Free PDF downloads. Shows prerequisites for each concept. |
| **Core Knowledge Sequence** | [coreknowledge.org/free-resource/core-knowledge-sequence](https://coreknowledge.org/free-resource/core-knowledge-sequence/) | **BEST for content** - Complete K-8 scope & sequence ALL subjects. Shows what knowledge builds on what. |
| **Illustrative Mathematics** | [illustrativemathematics.org/math-curriculum](https://illustrativemathematics.org/math-curriculum/) | Learning progressions research. Explains WHY sequences work. |
| **Open Up Resources** | [openupresources.org](https://openupresources.org/) | K-8 with implementation guides. HOW to teach sequences. |

### Using OER for AI Curriculum

```
For SEQUENCING (what comes before what):
  → Use Eureka Math module structure
  → Each module lists prerequisites explicitly

For CONTENT (what knowledge at what grade):
  → Use Core Knowledge Sequence
  → Knowledge Strand + Skills Strand

For WHY (cognitive development logic):
  → Use IM Learning Progressions
  → Explains the pedagogy behind sequences
```

### Key Insight: Real Curricula Have Already Solved Sequencing

Instead of inventing our own progression, we can use:
- **Eureka Math**: Module 1 → Module 2 → Module 3 per grade
- **Core Knowledge**: Grade K → Grade 1 → Grade 2 content spirals
- **Singapore Math**: Mastery-based progression (CPA approach)

These curricula represent decades of pedagogical research on optimal learning sequences.

---

## Hugging Face Resources (Teacher Advisor + Datasets)

Hugging Face provides both datasets AND pre-trained models that can serve as "teacher advisors."

### Top Datasets

| Dataset | URL | Why It Works |
|---------|-----|--------------|
| **TinyStories** | [huggingface.co/datasets/roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | **TOP PICK** - Stories using only 3-4 year old vocabulary. Perfect K-2 reading. |
| **FineWeb-Edu** | [huggingface.co/datasets/HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | **1.3T tokens** of educationally-filtered content. Quality scored 0-5. Teacher knowledge base. |
| **GSM8K** | [huggingface.co/datasets/openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | 8500 grade-school math problems with step-by-step solutions. |
| **Orca Math 200K** | [huggingface.co/datasets/microsoft/orca-math-word-problems-200k](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | 200K grade school math problems. Massive practice set. |
| **Elementary School Dataset** | [huggingface.co/datasets/FredTheOkGuy/elem-school-dataset](https://huggingface.co/datasets/FredTheOkGuy/elem-school-dataset) | Explicitly K-3. Structured Q&A for assessment. |
| **Children's Book Test** | [huggingface.co/datasets/cbt](https://huggingface.co/datasets/cbt) | Classic children's literature. Reading comprehension. |
| **MultiSim** | [huggingface.co/datasets/MichaelR207/MultiSim](https://huggingface.co/datasets/MichaelR207/MultiSim) | Text simplification pairs. Make content age-appropriate. |
| **Spell Correction** | [huggingface.co/datasets/torinriley/spell-correction](https://huggingface.co/datasets/torinriley/spell-correction) | Misspelled → correct pairs. How kids actually spell. |

### Teacher Advisor Models

| Model | URL | Role |
|-------|-----|------|
| **Merlyn Education Teacher Assistant** | [huggingface.co/MerlynMind/merlyn-education-teacher-assistant](https://huggingface.co/MerlynMind/merlyn-education-teacher-assistant) | Pre-trained classroom advisor. Suggests activities, explains concepts. |

### Integration Strategy

```
TEACHER ADVISOR (intermittent use):
  → FineWeb-Edu as knowledge base
  → Merlyn Teacher Assistant for responses
  → MultiSim to simplify explanations to grade level

STUDENT PRACTICE (continuous training):
  → Reading: TinyStories, Children's Book Test
  → Math: GSM8K, Orca Math 200K
  → Spelling: Spell Correction dataset

ASSESSMENT:
  → Elementary School Dataset (structured Q&A)
  → GSM8K with step-by-step solutions
```

### Key Insight: Teacher Models for "I Do" Phase

The Merlyn Teacher Assistant model can power the "I Do" demonstration phase:
- Explain concepts simply
- Generate worked examples
- Answer student questions

This lets the transformer student learn from a pre-trained "teacher" - the way human students learn from human teachers.

---

## Integration: The School Day

```
Morning:
  - Math (Singapore Math patterns)
  - Reading (word patterns)

Recess:
  - Physics Playground (Run or Walk, MotionSense)

Afternoon:
  - Music (rhythm patterns, beat tracking)
  - Art (Quick Draw strokes, shape recognition)
```

### Interleaving Strategy
- Don't finish one subject before starting another
- Rotate through subjects within each training session
- Let patterns from one subject reinforce another:
  - Math: sequences, counting
  - Music: rhythm sequences, counting beats
  - Physics: position sequences, counting oscillations
  - Art: stroke sequences, counting vertices

---

## Cross-Subject Pattern Connections

| Concept | Math | Physics | Music | Art |
|---------|------|---------|-------|-----|
| **Sequence** | Number series | Position over time | Note order | Stroke order |
| **Repetition** | Constant value | Oscillation | Beat/rhythm | Pattern/motif |
| **Change** | +1, -1 | Velocity | Tempo change | Line direction |
| **Cycle** | Modular arithmetic | Pendulum swing | Measure repeat | Shape closure |

---

## Next Steps

1. [ ] Create Kaggle notebook for Physics (Run or Walk + Pendulum)
2. [ ] Create Kaggle notebook for Music (Audio Guide + Beat Tracking)
3. [ ] Create Kaggle notebook for Art (Four Shapes + Quick Draw)
4. [ ] Design interleaved training loop that rotates subjects
5. [ ] Align difficulty across subjects (kindergarten → first grade → ...)
6. [ ] Test: Can a model learning from real data beat synthetic patterns?

---

## Key Insight

> "Why are we reinventing anything? Why not just use exactly what humans use?"

The curriculum shouldn't be designed from scratch. It should be **curated** from real human learning materials. Kaggle gives us access to the data; our job is to sequence it developmentally.

---

## Sources

### Physics
- [Run or Walk Data Analysis](https://kaggle.com/code/vmalyi/run-or-walk-data-analysis-and-visualization)
- [Human Activity Recognition with Mobile Sensing](https://kaggle.com/code/malekzadeh/human-activity-recognition-with-mobile-sensing)
- [Simple Pendulum Simulator](https://kaggle.com/code/bensalem14/simple-pendulum-simulator)

### Music
- [Beginner's Guide to Audio Data](https://kaggle.com/code/fizzbuzz/beginner-s-guide-to-audio-data)
- [Music Generation: Guide for Beginners](https://kaggle.com/code/naklecha/music-generation-guide-for-beginners)
- [Tempo Estimation and Beat Tracking](https://kaggle.com/code/enrcdamn/tempo-estimation-and-beat-tracking-pipeline)

### Art
- [Getting Started: Viewing Quick Draw Doodles](https://kaggle.com/code/inversion/getting-started-viewing-quick-draw-doodles-etc)
- [Let's Play With Quick Draw!!](https://kaggle.com/code/harunshimanto/lets-play-with-quick-draw)
- [Raw Stroke Sequences in 1D-CNN](https://kaggle.com/morrisb/raw-stroke-sequences-in-1d-cnn/data)
