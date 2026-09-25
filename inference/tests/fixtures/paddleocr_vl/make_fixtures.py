"""Renders the synthetic PaddleOCR-VL test images (made-up text, Noto Sans, OFL); rendered with Pillow 12.3."""
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Glyph rasterization differs slightly across Pillow/FreeType versions; the goldens belong to the committed PNGs.
FONT = os.environ.get("PADDLEOCR_VL_FIXTURE_FONT", "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf")
HERE = Path(__file__).parent
PAGE_SIZE = (640, 360)


def text_image(path, size, lines, font_size=28, margin=24, spacing=12):
    font = ImageFont.truetype(FONT, font_size)
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    y = margin
    for line in lines:
        draw.text((margin, y), line, font=font, fill="black")
        y += font_size + spacing
    image.save(path)


def table_image(path, rows, col_width=150, row_height=44, font_size=24, margin=10):
    font = ImageFont.truetype(FONT, font_size)
    width, height = 2 * margin + col_width * len(rows[0]), 2 * margin + row_height * len(rows)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            x, y = margin + c * col_width, margin + r * row_height
            draw.rectangle([x, y, x + col_width, y + row_height], outline="black", width=2)
            draw.text((x + 10, y + 8), cell, font=font, fill="black")
    image.save(path)


text_image(HERE / "ocr.png", (520, 80), ["The quick brown fox jumps 42 times."])
# Same size on purpose: the prefix-cache test needs byte-identical prompts that differ only in pixels.
text_image(HERE / "page_00.png", PAGE_SIZE, ["Weekly garden notes", "Tomatoes are ripening early.", "Water the beans twice."])
text_image(HERE / "page_01.png", PAGE_SIZE, ["Library opening hours", "Monday to Friday: 9 to 5.", "Closed on public holidays."])
table_image(HERE / "table.png", [["Fruit", "Colour", "Count"], ["Apple", "Red", "12"], ["Lemon", "Yellow", "7"]])
