#!/usr/bin/env python3
"""
Colour Lesson 04: Web UI Patterns
Common UI color patterns and accessibility considerations.
"""

from PIL import Image, ImageDraw, ImageFont

def create_ui_patterns():
    """Create common UI color patterns."""
    width, height = 700, 500
    img = Image.new('RGB', (width, height), '#0d1117')  # GitHub dark bg
    draw = ImageDraw.Draw(img)

    y = 20

    # Section 1: Status colors
    draw.text((20, y), 'STATUS COLORS', fill='#8b949e')
    y += 30

    statuses = [
        ('#238636', 'Success'),
        ('#f85149', 'Error'),
        ('#d29922', 'Warning'),
        ('#58a6ff', 'Info'),
        ('#8b949e', 'Neutral'),
    ]

    for i, (color, label) in enumerate(statuses):
        x = 20 + i * 130
        draw.rounded_rectangle([x, y, x + 110, y + 40], radius=6, fill=color)
        # Calculate text color for contrast
        draw.text((x + 55, y + 20), label, fill='white', anchor='mm')

    y += 70

    # Section 2: Dark mode palette
    draw.text((20, y), 'DARK MODE LAYERS (GitHub)', fill='#8b949e')
    y += 30

    dark_layers = [
        ('#010409', 'Canvas'),
        ('#0d1117', 'Default'),
        ('#161b22', 'Subtle'),
        ('#21262d', 'Muted'),
        ('#30363d', 'Border'),
    ]

    box_width = (width - 40) // len(dark_layers)
    for i, (color, label) in enumerate(dark_layers):
        x = 20 + i * box_width
        draw.rectangle([x, y, x + box_width - 4, y + 60], fill=color, outline='#30363d')
        draw.text((x + box_width // 2, y + 70), label, fill='#8b949e', anchor='mt')

    y += 110

    # Section 3: Light mode palette
    draw.text((20, y), 'LIGHT MODE LAYERS', fill='#8b949e')
    y += 30

    light_layers = [
        ('#ffffff', 'Canvas'),
        ('#f6f8fa', 'Subtle'),
        ('#eaeef2', 'Muted'),
        ('#d0d7de', 'Border'),
        ('#57606a', 'Text'),
    ]

    for i, (color, label) in enumerate(light_layers):
        x = 20 + i * box_width
        draw.rectangle([x, y, x + box_width - 4, y + 60], fill=color, outline='#d0d7de')
        text_color = '#24292f' if i < 4 else '#ffffff'
        draw.text((x + box_width // 2, y + 70), label, fill='#8b949e', anchor='mt')

    y += 110

    # Section 4: Contrast demo
    draw.text((20, y), 'CONTRAST: Accessible vs Inaccessible', fill='#8b949e')
    y += 30

    # Good contrast
    draw.rectangle([20, y, 340, y + 50], fill='#0d1117')
    draw.text((180, y + 25), 'Good: #ffffff on #0d1117', fill='#ffffff', anchor='mm')

    # Bad contrast
    draw.rectangle([360, y, 680, y + 50], fill='#21262d')
    draw.text((520, y + 25), 'Bad: #484f58 on #21262d', fill='#484f58', anchor='mm')

    return img

if __name__ == '__main__':
    img = create_ui_patterns()
    output_path = '/home/user/Emergence/lessons/colour/04_webui.png'
    img.save(output_path)
    print(f'Saved to {output_path}')
