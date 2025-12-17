"""
Visual Perception Learning Plan
===============================
A repeatable curriculum for building visual intuition through internal rendering.

Run this periodically to refresh/strengthen visual understanding.
Each module generates images and provides prompts for perception.

Usage:
    python visual_learning.py              # Run full curriculum
    python visual_learning.py color        # Run only color module
    python visual_learning.py layout       # Run only layout module
    python visual_learning.py --quick      # Quick refresher (key exercises only)
"""

import os
import sys
from PIL import Image, ImageDraw
import math
import colorsys

OUTPUT_DIR = "/tmp/visual_learning"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_and_print(img, name, perception_prompt):
    """Save image and print perception prompt."""
    path = f"{OUTPUT_DIR}/{name}.png"
    img.save(path)
    print(f"\n{'='*60}")
    print(f"Exercise: {name}")
    print(f"Saved to: {path}")
    print(f"\nPerception prompt: {perception_prompt}")
    print(f"{'='*60}")
    return path


# =============================================================================
# COLOR MODULE
# =============================================================================

class ColorModule:
    """Color theory exercises for building color intuition."""

    @staticmethod
    def hue_spectrum():
        """Full 360° hue spectrum - understand the color wheel."""
        img = Image.new('RGB', (360, 100), 'white')
        draw = ImageDraw.Draw(img)

        for x in range(360):
            # Convert hue to RGB (full saturation, full value)
            r, g, b = colorsys.hsv_to_rgb(x/360, 1.0, 1.0)
            color = (int(r*255), int(g*255), int(b*255))
            draw.line([(x, 0), (x, 100)], fill=color)

        return save_and_print(img, "01_hue_spectrum",
            "Observe: Where are the warm colors? Cool colors? "
            "Where does red become orange? Blue become purple? "
            "Notice the primary colors (red, yellow, blue) and where they sit.")

    @staticmethod
    def saturation_sweep():
        """Same hue, varying saturation - understand intensity."""
        img = Image.new('RGB', (400, 100), 'white')
        draw = ImageDraw.Draw(img)

        hue = 210/360  # Blue
        for i in range(10):
            sat = i / 9  # 0 to 1
            r, g, b = colorsys.hsv_to_rgb(hue, sat, 0.8)
            color = (int(r*255), int(g*255), int(b*255))
            draw.rectangle([i*40, 0, (i+1)*40, 100], fill=color)

        return save_and_print(img, "02_saturation_sweep",
            "Observe: Left is gray, right is vivid blue. "
            "Where does it start feeling 'colorful'? "
            "Which saturation level feels professional vs playful?")

    @staticmethod
    def value_sweep():
        """Same hue, varying brightness - understand lightness."""
        img = Image.new('RGB', (400, 100), 'white')
        draw = ImageDraw.Draw(img)

        hue = 210/360  # Blue
        for i in range(10):
            val = i / 9  # 0 to 1
            r, g, b = colorsys.hsv_to_rgb(hue, 0.7, val)
            color = (int(r*255), int(g*255), int(b*255))
            draw.rectangle([i*40, 0, (i+1)*40, 100], fill=color)

        return save_and_print(img, "03_value_sweep",
            "Observe: Left is nearly black, right is light. "
            "This is the same 'blue' - only brightness changes. "
            "Value creates the foundation of contrast and hierarchy.")

    @staticmethod
    def complementary_pairs():
        """Complementary colors side by side - see the vibration."""
        img = Image.new('RGB', (400, 150), 'white')
        draw = ImageDraw.Draw(img)

        pairs = [
            ((255, 0, 0), (0, 255, 255), "Red / Cyan"),
            ((0, 0, 255), (255, 255, 0), "Blue / Yellow"),
            ((255, 0, 255), (0, 255, 0), "Magenta / Green"),
        ]

        for i, (c1, c2, name) in enumerate(pairs):
            y = i * 50
            draw.rectangle([0, y, 200, y+50], fill=c1)
            draw.rectangle([200, y, 400, y+50], fill=c2)

        return save_and_print(img, "04_complementary_pairs",
            "Observe: Look at each boundary where colors meet. "
            "Do you see the 'vibration' or shimmer effect? "
            "Which pair has the most tension? Which is hardest to look at?")

    @staticmethod
    def analogous_harmony():
        """Analogous colors - smooth, harmonious transitions."""
        img = Image.new('RGB', (400, 100), 'white')
        draw = ImageDraw.Draw(img)

        base_hue = 200/360  # Start at blue
        for i in range(8):
            hue = (base_hue + (i * 0.05)) % 1.0  # 5% hue steps
            r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.8)
            color = (int(r*255), int(g*255), int(b*255))
            draw.rectangle([i*50, 0, (i+1)*50, 100], fill=color)

        return save_and_print(img, "05_analogous_harmony",
            "Observe: These are neighboring hues - blues and teals. "
            "Notice how smoothly they flow together. "
            "Compare this feeling to the complementary pairs above.")

    @staticmethod
    def triadic_balance():
        """Triadic colors with 60-30-10 distribution."""
        img = Image.new('RGB', (400, 150), 'white')
        draw = ImageDraw.Draw(img)

        # Equal distribution (chaotic)
        draw.rectangle([0, 0, 133, 75], fill=(255, 0, 0))
        draw.rectangle([133, 0, 266, 75], fill=(0, 0, 255))
        draw.rectangle([266, 0, 400, 75], fill=(255, 255, 0))

        # 60-30-10 distribution (balanced)
        draw.rectangle([0, 75, 240, 150], fill=(30, 60, 114))  # 60% dominant (muted blue)
        draw.rectangle([240, 75, 360, 150], fill=(255, 200, 87))  # 30% secondary (gold)
        draw.rectangle([360, 75, 400, 150], fill=(232, 65, 24))  # 10% accent (red)

        return save_and_print(img, "06_triadic_balance",
            "Observe: Top row is equal distribution - chaotic, no hierarchy. "
            "Bottom row is 60-30-10 - one color dominates, one supports, one accents. "
            "Which feels more professional? Which draws your eye to a focal point?")

    @staticmethod
    def semantic_colors():
        """Colors with meaning - error, warning, success, info."""
        img = Image.new('RGB', (400, 100), 'white')
        draw = ImageDraw.Draw(img)

        semantics = [
            ((220, 53, 69), "Error"),      # Red
            ((255, 193, 7), "Warning"),    # Yellow/Amber
            ((40, 167, 69), "Success"),    # Green
            ((0, 123, 255), "Info"),       # Blue
        ]

        for i, (color, label) in enumerate(semantics):
            draw.rectangle([i*100, 0, (i+1)*100, 100], fill=color)

        return save_and_print(img, "07_semantic_colors",
            "Observe: These colors carry instant meaning. "
            "Red = danger/error. Yellow = caution. Green = success. Blue = info. "
            "This is cultural convention - you understand it without being told.")

    @staticmethod
    def contrast_accessibility():
        """Text contrast for accessibility."""
        img = Image.new('RGB', (400, 200), 'white')
        draw = ImageDraw.Draw(img)

        # Good contrast
        draw.rectangle([0, 0, 200, 100], fill=(255, 255, 255))
        draw.rectangle([10, 30, 190, 70], fill=(33, 33, 33))  # Dark on light

        # Poor contrast
        draw.rectangle([200, 0, 400, 100], fill=(255, 255, 255))
        draw.rectangle([210, 30, 390, 70], fill=(200, 200, 200))  # Light on light

        # Dark mode - good
        draw.rectangle([0, 100, 200, 200], fill=(30, 30, 30))
        draw.rectangle([10, 130, 190, 170], fill=(224, 224, 224))  # Light on dark

        # Dark mode - poor
        draw.rectangle([200, 100, 400, 200], fill=(30, 30, 30))
        draw.rectangle([210, 130, 390, 170], fill=(80, 80, 80))  # Dark on dark

        return save_and_print(img, "08_contrast_accessibility",
            "Observe: Left column has good contrast - elements are clearly visible. "
            "Right column has poor contrast - elements fade into background. "
            "Accessibility isn't just compliance - it's basic visual communication.")


# =============================================================================
# LAYOUT MODULE
# =============================================================================

class LayoutModule:
    """Layout and composition exercises for building spatial intuition."""

    @staticmethod
    def proximity_grouping():
        """Proximity creates perceived groups."""
        img = Image.new('RGB', (400, 150), 'white')
        draw = ImageDraw.Draw(img)

        # Three groups of three circles
        groups = [(50, 75), (200, 75), (350, 75)]
        for gx, gy in groups:
            for dx in [-20, 0, 20]:
                draw.ellipse([gx+dx-10, gy-10, gx+dx+10, gy+10], fill='navy')

        return save_and_print(img, "10_proximity_grouping",
            "Observe: You see THREE GROUPS, not nine circles. "
            "The spacing between groups is larger than within groups. "
            "Proximity is the strongest grouping principle.")

    @staticmethod
    def similarity_grouping():
        """Similar elements are perceived as related."""
        img = Image.new('RGB', (400, 150), 'white')
        draw = ImageDraw.Draw(img)

        # Grid of circles - some blue, some orange
        for row in range(3):
            for col in range(8):
                x = 30 + col * 45
                y = 30 + row * 40
                # Diagonal pattern
                color = '#3498db' if (row + col) % 2 == 0 else '#e67e22'
                draw.ellipse([x-12, y-12, x+12, y+12], fill=color)

        return save_and_print(img, "11_similarity_grouping",
            "Observe: Despite equal spacing, you see TWO groups - blue and orange. "
            "Your brain connects same-colored elements across distance. "
            "Color similarity can organize scattered elements.")

    @staticmethod
    def closure():
        """The brain completes incomplete shapes."""
        img = Image.new('RGB', (400, 150), 'white')
        draw = ImageDraw.Draw(img)

        # Dotted circle - implied complete circle
        center = (100, 75)
        radius = 40
        for angle in range(0, 360, 30):
            x = center[0] + radius * math.cos(math.radians(angle))
            y = center[1] + radius * math.sin(math.radians(angle))
            draw.ellipse([x-5, y-5, x+5, y+5], fill='navy')

        # Four corners imply rectangle
        corners = [(220, 30), (380, 30), (220, 120), (380, 120)]
        for cx, cy in corners:
            # L-shaped corners
            draw.rectangle([cx, cy, cx+20, cy+5], fill='navy')
            draw.rectangle([cx, cy, cx+5, cy+20], fill='navy')

        return save_and_print(img, "12_closure",
            "Observe: Left - you see a CIRCLE, not 12 dots. "
            "Right - you see a RECTANGLE, not 4 corner pieces. "
            "Your brain completes the shapes automatically.")

    @staticmethod
    def figure_ground():
        """Distinguishing foreground from background."""
        img = Image.new('RGB', (400, 200), '#333333')
        draw = ImageDraw.Draw(img)

        # Modal overlay effect
        # Darkened background
        draw.rectangle([0, 0, 400, 200], fill='#1a1a1a')

        # Floating card (figure)
        draw.rectangle([100, 40, 300, 160], fill='white')
        draw.rectangle([120, 60, 280, 80], fill='#666')  # Title
        draw.rectangle([120, 90, 260, 100], fill='#999')  # Text
        draw.rectangle([120, 110, 240, 120], fill='#999')  # Text
        draw.rectangle([200, 135, 280, 150], fill='#3498db')  # Button

        return save_and_print(img, "13_figure_ground",
            "Observe: The white card 'floats' above the dark background. "
            "This is figure-ground separation - the modal pattern. "
            "The dimmed background pushes the card forward in visual space.")

    @staticmethod
    def visual_hierarchy_size():
        """Size creates importance hierarchy."""
        img = Image.new('RGB', (400, 200), 'white')
        draw = ImageDraw.Draw(img)

        # Different sized elements
        draw.rectangle([20, 20, 200, 80], fill='#2c3e50')   # Large - headline
        draw.rectangle([20, 95, 140, 115], fill='#7f8c8d')  # Medium - subhead
        draw.rectangle([20, 125, 300, 140], fill='#bdc3c7') # Lines - body
        draw.rectangle([20, 145, 280, 160], fill='#bdc3c7')
        draw.rectangle([20, 165, 320, 180], fill='#bdc3c7')

        return save_and_print(img, "14_hierarchy_size",
            "Observe: Your eye goes to the LARGEST element first. "
            "Then medium, then small. Size = importance. "
            "This is the foundation of typographic hierarchy.")

    @staticmethod
    def visual_hierarchy_color():
        """Color creates attention hierarchy."""
        img = Image.new('RGB', (400, 150), 'white')
        draw = ImageDraw.Draw(img)

        # Row of buttons - one stands out
        colors = ['#bdc3c7', '#bdc3c7', '#e74c3c', '#bdc3c7', '#bdc3c7']
        for i, color in enumerate(colors):
            x = 30 + i * 75
            draw.rectangle([x, 55, x+60, 95], fill=color)

        return save_and_print(img, "15_hierarchy_color",
            "Observe: Which button do you see FIRST? The red one. "
            "It's the same size as others, but color makes it pop. "
            "Use this sparingly - if everything is bright, nothing stands out.")

    @staticmethod
    def whitespace_impact():
        """Whitespace creates focus and premium feel."""
        img = Image.new('RGB', (400, 200), 'white')
        draw = ImageDraw.Draw(img)

        # Cramped layout (left)
        draw.rectangle([5, 5, 195, 195], outline='#ddd')
        for i in range(4):
            draw.rectangle([10, 10+i*45, 190, 50+i*45], fill='#95a5a6')

        # Spacious layout (right)
        draw.rectangle([205, 5, 395, 195], outline='#ddd')
        draw.rectangle([230, 50, 370, 90], fill='#95a5a6')
        draw.rectangle([230, 110, 370, 150], fill='#95a5a6')

        return save_and_print(img, "16_whitespace_impact",
            "Observe: Left feels cramped, cheap, overwhelming. "
            "Right feels spacious, premium, focused. "
            "Same amount of content - whitespace changes everything.")

    @staticmethod
    def grid_12_column():
        """12-column grid structure."""
        img = Image.new('RGB', (400, 200), 'white')
        draw = ImageDraw.Draw(img)

        # Draw 12 columns
        col_width = 400 / 12
        for i in range(12):
            x = i * col_width
            color = '#ecf0f1' if i % 2 == 0 else '#d5dbdb'
            draw.rectangle([x, 0, x + col_width, 200], fill=color)

        # Content spanning columns
        # Hero: 12 columns
        draw.rectangle([0, 10, 400, 50], fill='#3498db', outline='#2980b9')
        # Main: 8 columns, Sidebar: 4 columns
        draw.rectangle([0, 60, col_width*8, 150], fill='#2ecc71', outline='#27ae60')
        draw.rectangle([col_width*8, 60, 400, 150], fill='#9b59b6', outline='#8e44ad')
        # Footer: 12 columns
        draw.rectangle([0, 160, 400, 190], fill='#34495e', outline='#2c3e50')

        return save_and_print(img, "17_grid_12_column",
            "Observe: The gray stripes show the 12-column grid. "
            "Content aligns to column boundaries: 12, 8+4, 12. "
            "This creates consistent rhythm and alignment.")

    @staticmethod
    def f_pattern():
        """F-pattern reading flow."""
        img = Image.new('RGB', (400, 300), 'white')
        draw = ImageDraw.Draw(img)

        # Heatmap showing F-pattern attention
        # Top bar (highest attention)
        draw.rectangle([0, 0, 400, 40], fill='#e74c3c')
        # Left column (high attention)
        draw.rectangle([0, 40, 100, 300], fill='#e67e22')
        # Secondary scan
        draw.rectangle([100, 100, 400, 130], fill='#f39c12')
        # Rest (lower attention)
        draw.rectangle([100, 40, 400, 100], fill='#f1c40f')
        draw.rectangle([100, 130, 400, 300], fill='#f1c40f')

        return save_and_print(img, "18_f_pattern",
            "Observe: This shows where eyes go on text-heavy pages. "
            "Red = highest attention (top bar). Orange = left column. "
            "Yellow = secondary areas. Place important content in hot zones.")

    @staticmethod
    def z_pattern():
        """Z-pattern reading flow for visual pages."""
        img = Image.new('RGB', (400, 300), 'white')
        draw = ImageDraw.Draw(img)

        # Z-pattern path
        draw.rectangle([20, 20, 80, 50], fill='#e74c3c')   # Top-left (1)
        draw.rectangle([320, 20, 380, 50], fill='#e67e22') # Top-right (2)
        draw.rectangle([20, 230, 80, 260], fill='#f39c12') # Bottom-left (3)
        draw.rectangle([300, 230, 380, 280], fill='#27ae60') # Bottom-right (4) - CTA

        # Draw Z path
        draw.line([(50, 35), (350, 35)], fill='#95a5a6', width=2)
        draw.line([(350, 35), (50, 245)], fill='#95a5a6', width=2)
        draw.line([(50, 245), (340, 255)], fill='#95a5a6', width=2)

        return save_and_print(img, "19_z_pattern",
            "Observe: The eye follows a Z path on visual pages. "
            "Logo top-left → Nav top-right → Headline bottom-left → CTA bottom-right. "
            "The green rectangle is where your call-to-action should go.")

    @staticmethod
    def card_anatomy():
        """Standard card component structure."""
        img = Image.new('RGB', (400, 300), '#f5f5f5')
        draw = ImageDraw.Draw(img)

        # Card with shadow
        draw.rectangle([52, 52, 352, 252], fill='#ddd')  # Shadow
        draw.rectangle([50, 50, 350, 250], fill='white')

        # Image area (top 50%)
        draw.rectangle([50, 50, 350, 150], fill='#3498db')

        # Content area
        draw.rectangle([70, 165, 250, 180], fill='#2c3e50')  # Title
        draw.rectangle([70, 190, 330, 200], fill='#95a5a6')  # Description
        draw.rectangle([70, 205, 300, 215], fill='#95a5a6')  # Description

        # Button
        draw.rectangle([70, 225, 150, 240], fill='#e74c3c')

        return save_and_print(img, "20_card_anatomy",
            "Observe: Standard card structure - image top, content middle, action bottom. "
            "The shadow creates depth (figure-ground). "
            "Internal spacing creates hierarchy within the card.")


# =============================================================================
# MAIN
# =============================================================================

def run_color_module():
    """Run all color exercises."""
    print("\n" + "="*60)
    print("COLOR MODULE")
    print("="*60)

    ColorModule.hue_spectrum()
    ColorModule.saturation_sweep()
    ColorModule.value_sweep()
    ColorModule.complementary_pairs()
    ColorModule.analogous_harmony()
    ColorModule.triadic_balance()
    ColorModule.semantic_colors()
    ColorModule.contrast_accessibility()


def run_layout_module():
    """Run all layout exercises."""
    print("\n" + "="*60)
    print("LAYOUT MODULE")
    print("="*60)

    LayoutModule.proximity_grouping()
    LayoutModule.similarity_grouping()
    LayoutModule.closure()
    LayoutModule.figure_ground()
    LayoutModule.visual_hierarchy_size()
    LayoutModule.visual_hierarchy_color()
    LayoutModule.whitespace_impact()
    LayoutModule.grid_12_column()
    LayoutModule.f_pattern()
    LayoutModule.z_pattern()
    LayoutModule.card_anatomy()


def run_quick_refresher():
    """Quick refresher - key exercises only."""
    print("\n" + "="*60)
    print("QUICK REFRESHER")
    print("="*60)

    ColorModule.complementary_pairs()
    ColorModule.triadic_balance()
    LayoutModule.proximity_grouping()
    LayoutModule.visual_hierarchy_size()
    LayoutModule.whitespace_impact()


def main():
    ensure_output_dir()

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == 'color':
            run_color_module()
        elif arg == 'layout':
            run_layout_module()
        elif arg == '--quick':
            run_quick_refresher()
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python visual_learning.py [color|layout|--quick]")
    else:
        run_color_module()
        run_layout_module()

    print(f"\n\nAll images saved to: {OUTPUT_DIR}")
    print("View each image and read the perception prompts to build intuition.")


if __name__ == '__main__':
    main()
