#!/usr/bin/env python3
"""
Colour Lesson 03: Warm to Cool Gradients
Understanding colour temperature and its effect on perceived depth/energy.
"""

from PIL import Image, ImageDraw

def lerp_color(c1, c2, t):
    """Linear interpolation between two RGB tuples."""
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def hex_to_rgb(hex_color):
    """Convert hex to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def create_temperature_gradients():
    """Create gradients showing warm-cool relationships."""
    width, height = 600, 400
    img = Image.new('RGB', (width, height), '#1a1a1a')
    draw = ImageDraw.Draw(img)

    gradients = [
        # Warm to cool
        ('#FF6B35', '#004E89', 'Sunset: Orange → Deep Blue'),
        # Fire to ice
        ('#FF0000', '#00BFFF', 'Fire to Ice: Red → Sky Blue'),
        # Earth tones warm to cool
        ('#8B4513', '#2F4F4F', 'Earth: Saddle Brown → Dark Slate'),
        # Approachable gradient for UI
        ('#FF9F1C', '#2EC4B6', 'UI Friendly: Amber → Teal'),
    ]

    row_height = height // len(gradients)

    for i, (color1, color2, label) in enumerate(gradients):
        y0 = i * row_height
        y1 = y0 + row_height - 4

        rgb1 = hex_to_rgb(color1)
        rgb2 = hex_to_rgb(color2)

        # Draw gradient pixel by pixel (column by column)
        for x in range(width):
            t = x / width
            color = lerp_color(rgb1, rgb2, t)
            draw.line([(x, y0), (x, y1)], fill=color)

    return img

if __name__ == '__main__':
    img = create_temperature_gradients()
    output_path = '/home/user/Emergence/lessons/colour/03_warmcool.png'
    img.save(output_path)
    print(f'Saved to {output_path}')
