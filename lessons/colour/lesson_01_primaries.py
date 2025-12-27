#!/usr/bin/env python3
"""
Colour Lesson 01: Primary Colors
Generates a visual for perceiving the RGB primaries.
"""

from PIL import Image, ImageDraw, ImageFont

def create_primary_colors():
    """Create a visual showing the three RGB primaries with labels."""
    width, height = 600, 300
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    colors = [
        ('#FF0000', 'RED', 'Energy, urgency, passion, warnings'),
        ('#00FF00', 'GREEN', 'Growth, nature, success, go'),
        ('#0000FF', 'BLUE', 'Trust, calm, depth, professional'),
    ]

    box_width = width // 3

    for i, (hex_color, name, meaning) in enumerate(colors):
        x0 = i * box_width
        x1 = x0 + box_width

        # Draw color box (top 2/3)
        draw.rectangle([x0, 0, x1, height * 2 // 3], fill=hex_color)

        # Label area (bottom 1/3)
        draw.rectangle([x0, height * 2 // 3, x1, height], fill='#222222')

        # Add text
        text_x = x0 + box_width // 2
        draw.text((text_x, height * 2 // 3 + 10), name, fill='white', anchor='mt')
        draw.text((text_x, height * 2 // 3 + 35), hex_color, fill='#888888', anchor='mt')

    return img

if __name__ == '__main__':
    img = create_primary_colors()
    output_path = '/home/user/Emergence/lessons/colour/01_primaries.png'
    img.save(output_path)
    print(f'Saved to {output_path}')
