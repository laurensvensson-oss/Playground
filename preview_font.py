#!/usr/bin/env python3
"""Generate a preview image of the MyHandwriting font."""
from PIL import Image, ImageDraw, ImageFont

font_path = "font_output/MyHandwriting-Regular.ttf"
output_path = "font_output/preview.png"

img = Image.new("RGB", (900, 400), "white")
draw = ImageDraw.Draw(img)

try:
    font = ImageFont.truetype(font_path, 48)
except OSError:
    print(f"Could not load font from {font_path}")
    exit(1)

lines = [
    "ABCDEFGHIJKLM",
    "NOPQRSTUVWXYZ",
    "0123456789",
    "!@.,;#$%&*()?/-+",
]

y = 20
for line in lines:
    draw.text((30, y), line, fill="black", font=font)
    y += 80

img.save(output_path)
print(f"Preview saved to {output_path}")
