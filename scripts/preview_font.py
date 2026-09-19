"""Render a before/after font specimen using Pillow's FreeType renderer."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SAMPLES = {
    "latin": ["Hamburgefonts 0123456789", "The quick brown fox jumps",
              'over the lazy dog. "Aa" &!?', "HAMBURGEFONTS ABC XYZ",
              "agjpqy bdfhiklt .,;:-"],
    "cyrillic": ["Съешь ещё этих мягких", "французских булок, да выпей чаю.",
                 "АБВГДЕЁЖЗИЙКЛМНОПРСТУ", "ДЦЩ дцщ руф Ёё Йй 0123456789",
                 '.,;:- "Аа" &!?'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--language", choices=SAMPLES, default="latin")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    lines = SAMPLES[args.language]
    fonts = [[ImageFont.truetype(path, size) for size in (42, 64)]
             for path in (args.before, args.after)]
    column_width = int(max(font.getlength(line) for column in fonts
                           for font in column for line in lines)) + 60
    image = Image.new("RGB", (column_width * 2, 940), "white")
    draw = ImageDraw.Draw(image)
    for col, column in enumerate(fonts):
        x = 25 + col * column_width
        draw.text((x, 20), "BEFORE" if col == 0 else "AFTER", fill="black")
        for row, font in enumerate(column):
            for i, line in enumerate(lines):
                y = 100 + row * 430 + i * 78
                draw.line((x, y, x + column_width - 50, y), fill="#cbd8e5")
                draw.text((x, y), line, font=font, fill="black", anchor="ls")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


if __name__ == "__main__":
    main()
