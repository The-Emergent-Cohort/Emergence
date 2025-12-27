#!/usr/bin/env python3
"""
Colour Lesson 02: Complementary Colors
Colors opposite each other on the color wheel - maximum contrast.
"""

from PIL import Image, ImageDraw

def create_complementary_pairs():
    """Show complementary color pairs side by side."""
    width, height = 600, 400
    img = Image.new('RGB', (width, height), '#1a1a1a')
    draw = ImageDraw.Draw(img)

    pairs = [
        ('#FF0000', '#00FFFF', 'Red / Cyan'),
        ('#00FF00', '#FF00FF', 'Green / Magenta'),
        ('#0000FF', '#FFFF00', 'Blue / Yellow'),
    ]

    row_height = height // 3
    box_width = width // 2

    for i, (color1, color2, label) in enumerate(pairs):
        y0 = i * row_height
        y1 = y0 + row_height - 4  # small gap

        # Left color
        draw.rectangle([0, y0, box_width - 2, y1], fill=color1)
        # Right color
        draw.rectangle([box_width + 2, y0, width, y1], fill=color2)

        # Center label on dark strip
        draw.rectangle([box_width - 2, y0, box_width + 2, y1], fill='#1a1a1a')

    return img

if __name__ == '__main__':
    img = create_complementary_pairs()
    output_path = '/home/user/Emergence/lessons/colour/02_complements.png'
    img.save(output_path)
    print(f'Saved to {output_path}')
