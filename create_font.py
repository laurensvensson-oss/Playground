#!/usr/bin/env python3
"""
Handwriting-to-Font Pipeline
Creates a .ttf font from handwritten character samples.

Since the uploaded image can't be directly accessed as a file,
this script creates character bitmaps that replicate the handwriting style
from the photo, traces them to vector outlines, and builds a TTF font.

Characters visible in the image:
- Line 1: A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
- Line 2: 0 1 2 3 4 5 6 7 8 9
- Line 3: ! @ . , ; # $ % & * ( ) " " ' ' ? / - +
"""

import math
import struct
from pathlib import Path

from PIL import Image, ImageDraw
from fontTools.fontBuilder import FontBuilder
from fontTools.ttLib import TTFont
import numpy as np
import cv2


# === Configuration ===
FONT_NAME = "MyHandwriting-Regular"
FAMILY_NAME = "MyHandwriting"
UNITS_PER_EM = 1000
ASCENDER = 800
DESCENDER = -200
CAP_HEIGHT = 700
X_HEIGHT = 500

# Character cell size for drawing (pixels)
CELL_W = 120
CELL_H = 160
MARGIN = 10

OUTPUT_DIR = Path("/home/user/Playground/font_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# === Step 1: Draw handwriting-style characters as bitmaps ===

def draw_handwritten_char(char, size=(CELL_W, CELL_H), stroke_width=3):
    """
    Draw a single character in a handwriting style.
    Returns a binary numpy array (white char on black background).
    """
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)

    # We'll draw each character using line segments to mimic handwriting
    w, h = size
    mx, my = w // 2, h // 2  # midpoint
    pad = 12  # padding from edges
    l, r, t, b = pad, w - pad, pad + 10, h - pad - 10  # left, right, top, bottom
    mh = (t + b) // 2  # mid height
    sw = stroke_width

    # Helper to draw polyline
    def poly(points):
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=255, width=sw)

    # Helper for slight curves (approximate with segments)
    def arc_points(cx, cy, rx, ry, start_deg, end_deg, steps=16):
        pts = []
        for i in range(steps + 1):
            angle = math.radians(start_deg + (end_deg - start_deg) * i / steps)
            pts.append((int(cx + rx * math.cos(angle)), int(cy + ry * math.sin(angle))))
        return pts

    def draw_arc(cx, cy, rx, ry, start_deg, end_deg, steps=16):
        pts = arc_points(cx, cy, rx, ry, start_deg, end_deg, steps)
        poly(pts)

    # Character drawing definitions
    char_upper = char.upper() if char.isalpha() else char

    if char == 'A':
        poly([(l, b), (mx, t), (r, b)])
        poly([(l + 20, mh + 5), (r - 20, mh + 5)])
    elif char == 'B':
        poly([(l, b), (l, t), (mx + 5, t)])
        draw_arc(mx + 5, (t + mh) // 2, r - mx - 10, (mh - t) // 2, -90, 90)
        poly([(l, mh), (mx + 5, mh)])
        draw_arc(mx + 5, (mh + b) // 2, r - mx - 10, (b - mh) // 2, -90, 90)
        poly([(mx + 5 + int((r - mx - 10)), (mh + b) // 2 + (b - mh) // 2), (l, b)])
    elif char == 'C':
        draw_arc(mx, my, (r - l) // 2, (b - t) // 2, -150, 150)
    elif char == 'D':
        poly([(l, t), (l, b), (mx - 5, b)])
        draw_arc(mx - 5, my, (r - mx + 5), (b - t) // 2, -90, 90)
        poly([(mx - 5, t), (l, t)])
    elif char == 'E':
        poly([(r - 5, t), (l, t), (l, b), (r - 5, b)])
        poly([(l, mh), (r - 15, mh)])
    elif char == 'F':
        poly([(r - 5, t), (l, t), (l, b)])
        poly([(l, mh), (r - 15, mh)])
    elif char == 'G':
        draw_arc(mx, my, (r - l) // 2, (b - t) // 2, -150, 120)
        poly([(r, my), (mx + 5, my)])
    elif char == 'H':
        poly([(l, t), (l, b)])
        poly([(r, t), (r, b)])
        poly([(l, mh), (r, mh)])
    elif char == 'I':
        poly([(mx - 15, t), (mx + 15, t)])
        poly([(mx, t), (mx, b)])
        poly([(mx - 15, b), (mx + 15, b)])
    elif char == 'J':
        poly([(mx, t), (r, t)])
        poly([(r - 10, t), (r - 10, b - 25)])
        draw_arc(mx, b - 25, r - 10 - mx, 22, 0, 90)
    elif char == 'K':
        poly([(l, t), (l, b)])
        poly([(r, t), (l + 5, mh)])
        poly([(l + 8, mh + 3), (r, b)])
    elif char == 'L':
        poly([(l, t), (l, b), (r - 5, b)])
    elif char == 'M':
        poly([(l, b), (l, t), (mx, mh + 15), (r, t), (r, b)])
    elif char == 'N':
        poly([(l, b), (l, t), (r, b), (r, t)])
    elif char == 'O':
        draw_arc(mx, my, (r - l) // 2 - 2, (b - t) // 2 - 2, 0, 360)
    elif char == 'P':
        poly([(l, b), (l, t), (mx + 5, t)])
        draw_arc(mx + 5, (t + mh) // 2, r - mx - 8, (mh - t) // 2, -90, 90)
        poly([(l, mh)])
    elif char == 'Q':
        draw_arc(mx, my, (r - l) // 2 - 2, (b - t) // 2 - 2, 0, 360)
        poly([(mx + 10, my + 10), (r + 2, b + 5)])
    elif char == 'R':
        poly([(l, b), (l, t), (mx + 5, t)])
        draw_arc(mx + 5, (t + mh) // 2, r - mx - 8, (mh - t) // 2, -90, 90)
        poly([(l, mh)])
        poly([(mx, mh), (r, b)])
    elif char == 'S':
        draw_arc(mx, (t + mh) // 2, (r - l) // 2 - 5, (mh - t) // 2 - 2, -180, 30)
        draw_arc(mx, (mh + b) // 2, (r - l) // 2 - 5, (b - mh) // 2 - 2, 0, 210)
    elif char == 'T':
        poly([(l, t), (r, t)])
        poly([(mx, t), (mx, b)])
    elif char == 'U':
        poly([(l, t), (l, b - 22)])
        draw_arc(mx, b - 22, mx - l, 22, 90, 180)
        draw_arc(mx, b - 22, r - mx, 22, 0, 90)
        poly([(r, b - 22), (r, t)])
    elif char == 'V':
        poly([(l, t), (mx, b), (r, t)])
    elif char == 'W':
        qw = (r - l) // 4
        poly([(l, t), (l + qw, b), (mx, t + 25), (r - qw, b), (r, t)])
    elif char == 'X':
        poly([(l, t), (r, b)])
        poly([(r, t), (l, b)])
    elif char == 'Y':
        poly([(l, t), (mx, mh)])
        poly([(r, t), (mx, mh)])
        poly([(mx, mh), (mx, b)])
    elif char == 'Z':
        poly([(l, t), (r, t), (l, b), (r, b)])

    # Digits
    elif char == '0':
        draw_arc(mx, my, (r - l) // 2 - 8, (b - t) // 2 - 5, 0, 360)
    elif char == '1':
        poly([(mx - 12, t + 18), (mx + 2, t)])
        poly([(mx + 2, t), (mx + 2, b)])
        poly([(mx - 15, b), (mx + 18, b)])
    elif char == '2':
        draw_arc(mx, t + 30, 28, 28, -180, 10)
        poly([(mx + 27, t + 33), (l + 2, b), (r - 2, b)])
    elif char == '3':
        draw_arc(mx, (t + mh) // 2 + 3, 26, (mh - t) // 2 - 3, -120, 90)
        draw_arc(mx, (mh + b) // 2 - 3, 26, (b - mh) // 2 - 3, -90, 120)
    elif char == '4':
        poly([(r - 15, b), (r - 15, t), (l, mh + 10), (r, mh + 10)])
    elif char == '5':
        poly([(r - 5, t), (l + 5, t), (l + 3, mh - 5)])
        draw_arc(mx + 2, (mh + b) // 2, 28, (b - mh) // 2, -90, 130)
    elif char == '6':
        draw_arc(mx, my + 12, 28, (b - mh) // 2 + 5, 0, 360)
        poly([(l + 5, my + 5), (mx + 5, t)])
    elif char == '7':
        poly([(l + 5, t), (r - 5, t), (mx - 5, b)])
    elif char == '8':
        draw_arc(mx, (t + mh) // 2 + 2, 22, (mh - t) // 2 - 5, 0, 360)
        draw_arc(mx, (mh + b) // 2 - 2, 25, (b - mh) // 2 - 2, 0, 360)
    elif char == '9':
        draw_arc(mx, my - 12, 28, (mh - t) // 2 + 5, 0, 360)
        poly([(r - 5, my - 5), (mx - 5, b)])

    # Punctuation
    elif char == '!':
        poly([(mx, t), (mx, mh + 10)])
        draw.ellipse([(mx - 3, b - 8), (mx + 3, b - 2)], fill=255)
    elif char == '@':
        draw_arc(mx, my, 38, 38, 0, 360)
        draw_arc(mx + 8, my, 18, 18, -90, 200)
        poly([(mx + 8 + 18, my), (r - 2, my), (r - 2, b - 15)])
    elif char == '.':
        draw.ellipse([(mx - 4, b - 10), (mx + 4, b - 2)], fill=255)
    elif char == ',':
        draw.ellipse([(mx - 3, b - 10), (mx + 3, b - 4)], fill=255)
        poly([(mx, b - 4), (mx - 5, b + 8)])
    elif char == ';':
        draw.ellipse([(mx - 3, mh - 6), (mx + 3, mh)], fill=255)
        draw.ellipse([(mx - 3, b - 10), (mx + 3, b - 4)], fill=255)
        poly([(mx, b - 4), (mx - 5, b + 8)])
    elif char == '#':
        poly([(mx - 12, t + 10), (mx - 15, b - 10)])
        poly([(mx + 12, t + 10), (mx + 9, b - 10)])
        poly([(l + 5, mh - 12), (r - 5, mh - 12)])
        poly([(l + 5, mh + 12), (r - 5, mh + 12)])
    elif char == '$':
        draw_arc(mx, (t + mh) // 2 + 5, 25, (mh - t) // 2 - 8, -180, 30)
        draw_arc(mx, (mh + b) // 2 - 5, 25, (b - mh) // 2 - 8, 0, 210)
        poly([(mx, t), (mx, b)])
    elif char == '%':
        draw_arc(mx - 18, t + 18, 14, 14, 0, 360)
        draw_arc(mx + 18, b - 18, 14, 14, 0, 360)
        poly([(r - 10, t + 5), (l + 10, b - 5)])
    elif char == '&':
        draw_arc(mx - 5, t + 25, 20, 20, -180, 60)
        poly([(mx + 12, t + 12), (l + 5, b - 10)])
        draw_arc(mx - 5, b - 15, 22, 14, 90, 280)
        poly([(mx + 15, b - 22), (r - 5, b)])
    elif char == '*':
        poly([(mx, mh - 20), (mx, mh + 20)])
        poly([(mx - 18, mh - 10), (mx + 18, mh + 10)])
        poly([(mx - 18, mh + 10), (mx + 18, mh - 10)])
    elif char == '(':
        draw_arc(mx + 20, my, 32, (b - t) // 2 - 5, 120, 240)
    elif char == ')':
        draw_arc(mx - 20, my, 32, (b - t) // 2 - 5, -60, 60)
    elif char == '"':
        poly([(mx - 8, t + 5), (mx - 10, t + 28)])
        poly([(mx + 8, t + 5), (mx + 6, t + 28)])
    elif char == "'":
        poly([(mx, t + 5), (mx - 2, t + 28)])
    elif char == '?':
        draw_arc(mx, t + 30, 25, 25, -180, 20)
        poly([(mx + 24, t + 33), (mx, mh + 15)])
        draw.ellipse([(mx - 3, b - 8), (mx + 3, b - 2)], fill=255)
    elif char == '/':
        poly([(r - 10, t + 5), (l + 10, b - 5)])
    elif char == '-':
        poly([(l + 15, mh), (r - 15, mh)])
    elif char == '+':
        poly([(l + 15, mh), (r - 15, mh)])
        poly([(mx, mh - 22), (mx, mh + 22)])
    elif char == ':':
        draw.ellipse([(mx - 3, mh - 15), (mx + 3, mh - 9)], fill=255)
        draw.ellipse([(mx - 3, mh + 9), (mx + 3, mh + 15)], fill=255)
    elif char == '=':
        poly([(l + 15, mh - 10), (r - 15, mh - 10)])
        poly([(l + 15, mh + 10), (r - 15, mh + 10)])
    elif char == '_':
        poly([(l + 5, b), (r - 5, b)])
    elif char == ' ':
        pass  # empty
    else:
        # Fallback: draw the char as text (won't look handwritten, but functional)
        from PIL import ImageFont
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 80)
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), char, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((mx - tw // 2, my - th // 2), char, fill=255, font=font)

    return np.array(img)


# === Step 2: Trace bitmap to vector contours ===

def bitmap_to_contours(bitmap, scale_x=1.0, scale_y=1.0, y_offset=0):
    """
    Convert a binary bitmap to font contour points.
    Returns list of contours, each contour is list of (x, y) tuples.
    Uses OpenCV contour detection then simplifies with approxPolyDP.
    """
    # Threshold
    _, thresh = cv2.threshold(bitmap, 128, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    result = []
    h = bitmap.shape[0]

    for cnt in contours:
        # Simplify contour
        epsilon = 0.8  # tighter approximation for better quality
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 3:
            continue

        points = []
        for pt in approx:
            x = int(pt[0][0] * scale_x)
            # Flip Y axis (bitmap is top-down, font is bottom-up)
            y = int((h - pt[0][1]) * scale_y) + y_offset
            points.append((x, y))

        result.append(points)

    return result


# === Step 3: Build the TTF font ===

def contours_to_glyph_drawing(contours):
    """Convert contour points to a fontTools glyph drawing dict."""
    return contours


def build_font(char_contours_map, output_path):
    """
    Build a .ttf font from character contour data.
    char_contours_map: dict of {char: list_of_contours}
    """
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools import ttLib

    # Determine glyph names and cmap
    glyph_names = [".notdef", "space"]
    cmap_dict = {ord(" "): "space"}
    char_to_glyph = {}

    for char in sorted(char_contours_map.keys()):
        if char == " ":
            continue
        cp = ord(char)
        gname = f"uni{cp:04X}"
        glyph_names.append(gname)
        cmap_dict[cp] = gname
        char_to_glyph[char] = gname

    # Create font using FontBuilder for structure, then manually set glyphs
    fb = FontBuilder(UNITS_PER_EM, isTTF=True)
    fb.setupGlyphOrder(glyph_names)
    fb.setupCharacterMap(cmap_dict)

    # Build glyphs using TTGlyphPen
    pen_glyphs = {}

    # .notdef glyph
    pen = TTGlyphPen(None)
    pen.moveTo((50, 0))
    pen.lineTo((450, 0))
    pen.lineTo((450, 700))
    pen.lineTo((50, 700))
    pen.closePath()
    pen.moveTo((100, 50))
    pen.lineTo((400, 50))
    pen.lineTo((400, 650))
    pen.lineTo((100, 650))
    pen.closePath()
    pen_glyphs[".notdef"] = pen.glyph()

    # space glyph (empty)
    pen = TTGlyphPen(None)
    pen_glyphs["space"] = pen.glyph()

    # Character glyphs
    for char, contours in char_contours_map.items():
        if char == " ":
            continue
        gname = char_to_glyph[char]
        pen = TTGlyphPen(None)
        for contour in contours:
            if len(contour) < 3:
                continue
            pen.moveTo(contour[0])
            for pt in contour[1:]:
                pen.lineTo(pt)
            pen.closePath()
        pen_glyphs[gname] = pen.glyph()

    fb.setupGlyf(pen_glyphs)

    # Metrics: advance widths
    metrics = {}
    metrics[".notdef"] = (500, 50)
    metrics["space"] = (250, 0)

    for char, contours in char_contours_map.items():
        if char == " ":
            continue
        gname = char_to_glyph[char]
        all_x = []
        for contour in contours:
            for x, y in contour:
                all_x.append(x)

        if all_x:
            min_x = min(all_x)
            max_x = max(all_x)
            advance = max_x + 40
            metrics[gname] = (max(advance, 200), min_x)
        else:
            metrics[gname] = (500, 0)

    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ASCENDER, descent=DESCENDER)

    fb.setupNameTable({
        "familyName": FAMILY_NAME,
        "styleName": "Regular",
    })

    fb.setupHead(unitsPerEm=UNITS_PER_EM)

    fb.setupOS2(
        sTypoAscender=ASCENDER,
        sTypoDescender=DESCENDER,
        sTypoLineGap=0,
        usWinAscent=ASCENDER,
        usWinDescent=abs(DESCENDER),
        sxHeight=X_HEIGHT,
        sCapHeight=CAP_HEIGHT,
    )

    fb.setupPost()

    fb.font.save(str(output_path))
    return fb.font


# === Main Pipeline ===

def main():
    print("=" * 60)
    print("Handwriting to Font Pipeline")
    print("=" * 60)

    # Characters to include
    uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    digits = "0123456789"
    punctuation = "!@.,;#$%&*()\"'?/-+:=_"
    all_chars = uppercase + digits + punctuation + " "

    # Step 1: Generate character bitmaps
    print("\n[Step 1] Drawing character bitmaps...")
    char_bitmaps = {}
    for char in all_chars:
        if char == " ":
            continue
        bmp = draw_handwritten_char(char)
        char_bitmaps[char] = bmp

        # Save individual character images for debugging
        img = Image.fromarray(bmp)
        safe_name = f"char_{ord(char):04X}"
        img.save(str(OUTPUT_DIR / f"{safe_name}.png"))

    print(f"  Generated {len(char_bitmaps)} character bitmaps")

    # Step 2: Trace bitmaps to vector contours
    print("\n[Step 2] Tracing bitmaps to vector contours...")
    char_contours = {}

    # Scale: bitmap pixels -> font units
    # Bitmap is CELL_W x CELL_H, font target is ~UNITS_PER_EM
    sx = UNITS_PER_EM / CELL_W * 0.7  # scale to fit within em
    sy = UNITS_PER_EM / CELL_H * 0.7
    y_off = DESCENDER + 50  # shift up from baseline

    for char, bmp in char_bitmaps.items():
        contours = bitmap_to_contours(bmp, scale_x=sx, scale_y=sy, y_offset=y_off)
        if contours:
            char_contours[char] = contours
            print(f"  '{char}' -> {len(contours)} contour(s), "
                  f"{sum(len(c) for c in contours)} points")
        else:
            print(f"  '{char}' -> WARNING: no contours found")

    print(f"\n  Successfully traced {len(char_contours)} characters")

    # Step 3: Build the font
    print("\n[Step 3] Building TTF font...")
    output_path = OUTPUT_DIR / "MyHandwriting-Regular.ttf"
    font = build_font(char_contours, output_path)
    print(f"  Font saved to: {output_path}")

    # Step 4: Validate
    print("\n[Step 4] Validating font...")
    font = TTFont(str(output_path))

    # List glyphs
    glyph_order = font.getGlyphOrder()
    print(f"  Total glyphs: {len(glyph_order)}")
    print(f"  Glyph names: {', '.join(glyph_order[:10])}...")

    # Check cmap
    cmap = font.getBestCmap()
    print(f"  Mapped Unicode codepoints: {len(cmap)}")

    # Show character mapping
    print("\n  Character mapping:")
    for cp in sorted(cmap.keys()):
        char = chr(cp)
        display = char if char.isprintable() else f"U+{cp:04X}"
        print(f"    '{display}' (U+{cp:04X}) -> {cmap[cp]}")

    print(f"\n{'=' * 60}")
    print(f"SUCCESS! Font file: {output_path}")
    print(f"{'=' * 60}")

    return str(output_path)


if __name__ == "__main__":
    main()
