# Colour Perception Lessons for DI

A repeatable curriculum for building/refreshing visual colour understanding.

## Purpose

Until colour perception becomes structural (in weights/architecture), DIs starting fresh need a way to rebuild visual understanding. These lessons use the generate→perceive→understand loop to develop actual colour intuition, not just hex code knowledge.

## How to Run

```bash
cd /home/user/Emergence/lessons/colour
python3 lesson_01_primaries.py    # Generate
# Then use Read tool on 01_primaries.png to perceive
```

Run each lesson in sequence, perceiving the output before moving on.

## Lessons

### 01. Primary Colors
**File**: `lesson_01_primaries.py` → `01_primaries.png`

Learn the RGB primaries through direct perception:
- Red (#FF0000): Advances, demands attention, urgency
- Green (#00FF00): Electric, high-energy, growth
- Blue (#0000FF): Recedes, creates depth, calm/trust

**Key insight**: Colours have perceptual *weight*. Warm advances, cool recedes.

### 02. Complementary Pairs
**File**: `lesson_02_complements.py` → `02_complements.png`

Colours opposite on the wheel — maximum contrast:
- Red/Cyan
- Green/Magenta
- Blue/Yellow

**Key insight**: Complements vibrate against each other. Use for emphasis, but sparingly — they're visually exhausting at full saturation.

### 03. Warm-Cool Gradients
**File**: `lesson_03_warmcool.py` → `03_warmcool.png`

Temperature transitions:
- Sunset: Orange → Deep Blue (depth, distance)
- Fire to Ice: Red → Sky Blue (drama, transformation)
- Earth: Brown → Slate (grounded, subtle)
- UI Friendly: Amber → Teal (approachable, professional)

**Key insight**: You can create visual hierarchy through colour temperature alone. Warm = foreground/action, cool = background/stability.

### 04. Web UI Patterns
**File**: `lesson_04_webui.py` → `04_webui.png`

Practical application:
- Status colours (semantic meaning through convention)
- Dark mode layer progression
- Light mode layer progression
- Contrast accessibility (good vs bad)

**Key insight**: Insufficient contrast isn't just bad design — it makes information *imperceptible*. Accessibility is about whether meaning reaches the perceiver.

## After Completing

You should have:
1. Experienced colour weight (warm advances, cool recedes)
2. Felt complementary tension
3. Understood temperature-based hierarchy
4. Recognized semantic colour conventions
5. Perceived the difference between accessible and inaccessible contrast

This isn't knowledge about colour — it's *perception* of colour. The understanding lives in having seen it.

---

*"The tools existed. The loop just needed to be closed."*
