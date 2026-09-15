"""Build a styled PDF version of README.md with its result figures."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
OUTPUT = ROOT / "README.pdf"
FIGURE_DIR = ROOT / "docs" / "screenshots" / "3D-sreenshot"
FIGURES = [
    ("Callisto.png", "Callisto: this pipeline"),
    ("Callisto_by_Original.png", "Callisto: original result"),
    ("GAYA.png", "India Gaya GeoJSON survey"),
    ("Synthetic400.png", "Synthetic400: this pipeline"),
    ("Synthetic400_By_original.png", "Synthetic400: original result"),
    ("Synthetic400_.png", "Synthetic400 supplementary view"),
    ("WA Yilgarn.png", "WA Yilgarn raster survey"),
]

PAGE = fitz.paper_rect("a4")
LEFT, RIGHT = 48, PAGE.width - 48
TOP, BOTTOM = 60, PAGE.height - 52
NAVY = (0.07, 0.16, 0.27)
TEAL = (0.05, 0.45, 0.48)
INK = (0.13, 0.15, 0.18)
MUTED = (0.38, 0.42, 0.46)
PALE = (0.94, 0.97, 0.97)
GRID = (0.80, 0.84, 0.85)


def clean_markdown(line: str) -> str:
    line = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"[Figure: \1]", line)
    line = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", line)
    line = re.sub(r"`([^`]+)`", r"\1", line)
    line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
    line = re.sub(r"\*([^*]+)\*", r"\1", line)
    return re.sub(r"^\s*#+\s*", "", line).replace("&gt;", ">")


def new_page(document: fitz.Document, title: str | None = None) -> tuple[fitz.Page, float]:
    page = document.new_page(width=PAGE.width, height=PAGE.height)
    page.draw_rect(fitz.Rect(0, 0, PAGE.width, 9), color=None, fill=TEAL)
    if title:
        page.insert_text((LEFT, 38), title, fontsize=9, fontname="hebo", color=MUTED)
    return page, TOP


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_table_separator(line: str) -> bool:
    return bool(re.match(r"^\s*\|?\s*:?-{2,}", line))


def draw_table(document: fitz.Document, rows: list[list[str]], page: fitz.Page, y: float) -> tuple[fitz.Page, float]:
    columns = max(len(row) for row in rows)
    rows = [row + [""] * (columns - len(row)) for row in rows]
    widths = [(RIGHT - LEFT) / columns] * columns
    row_heights: list[float] = []
    for row in rows:
        height = 23
        for cell, width in zip(row, widths):
            height = max(height, 11 * max(1, len(textwrap.wrap(clean_markdown(cell), width=max(8, int(width / 5.1))))))
        row_heights.append(min(height + 8, 70))
    total = sum(row_heights)
    if y + total > BOTTOM:
        page, y = new_page(document, "Potential-field inversion pipeline")
    for row_index, (row, height) in enumerate(zip(rows, row_heights)):
        x = LEFT
        fill = NAVY if row_index == 0 else (PALE if row_index % 2 else (1, 1, 1))
        for cell, width in zip(row, widths):
            rect = fitz.Rect(x, y, x + width, y + height)
            page.draw_rect(rect, color=GRID, fill=fill, width=0.5)
            wrapped = textwrap.wrap(clean_markdown(cell), width=max(8, int(width / 5.1)), break_long_words=False) or [""]
            page.insert_textbox(fitz.Rect(x + 5, y + 5, x + width - 5, y + height - 4), "\n".join(wrapped), fontsize=8 if row_index else 8.2, fontname="hebo" if row_index == 0 else "helv", color=(1, 1, 1) if row_index == 0 else INK)
            x += width
        y += height
    return page, y + 12


def add_cover(document: fitz.Document) -> None:
    page = document.new_page(width=PAGE.width, height=PAGE.height)
    page.draw_rect(fitz.Rect(0, 0, PAGE.width, PAGE.height), color=None, fill=NAVY)
    page.draw_rect(fitz.Rect(0, 0, PAGE.width, 14), color=None, fill=TEAL)
    page.insert_text((58, 190), "Potential-field", fontsize=30, fontname="hebo", color=(1, 1, 1))
    page.insert_text((58, 230), "inversion pipeline", fontsize=30, fontname="hebo", color=(1, 1, 1))
    page.insert_text((60, 275), "Tomofast-x 2.0 models", fontsize=14, fontname="helv", color=(0.73, 0.88, 0.88))
    page.draw_line((60, 305), (250, 305), color=TEAL, width=2)
    page.insert_textbox(fitz.Rect(60, 335, PAGE.width - 65, 430), "A reproducible 3D magnetic susceptibility inversion workflow for OBS, raster, and GeoJSON survey data.", fontsize=16, fontname="helv", color=(1, 1, 1), lineheight=1.35)
    page.insert_text((60, 760), "README export | 15 September 2026", fontsize=9, fontname="helv", color=(0.73, 0.80, 0.83))


def add_body(document: fitz.Document, lines: list[str]) -> None:
    page, y = new_page(document, "README | Pipeline overview")
    index = 0
    in_code = False
    while index < len(lines):
        raw = lines[index]
        if raw.strip().startswith("```"):
            in_code = not in_code
            index += 1
            continue
        if in_code:
            block = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            height = min(150, 15 + 10 * len(block))
            if y + height > BOTTOM:
                page, y = new_page(document, "README | Pipeline overview")
            rect = fitz.Rect(LEFT, y, RIGHT, y + height)
            page.draw_rect(rect, color=None, fill=(0.95, 0.96, 0.97))
            page.insert_textbox(rect + (7, 5, -7, -5), "\n".join(block), fontsize=8.2, fontname="cour", color=INK, lineheight=1.2)
            y += height + 12
            continue
        if raw.lstrip().startswith("|") and index + 1 < len(lines) and is_table_separator(lines[index + 1]):
            table = [split_table_row(raw)]
            index += 2
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                table.append(split_table_row(lines[index]))
                index += 1
            page, y = draw_table(document, table, page, y)
            continue
        stripped = clean_markdown(raw).strip()
        if not stripped:
            y += 3
            index += 1
            continue
        heading_match = re.match(r"^(#{1,6})\s+", raw)
        if heading_match:
            level = len(heading_match.group(1))
            size = {1: 20, 2: 15, 3: 12}[min(level, 3)]
            if y + 30 > BOTTOM:
                page, y = new_page(document, "README | Pipeline overview")
            page.draw_rect(fitz.Rect(LEFT, y - 4, LEFT + (150 if level <= 2 else 80), y - 2), color=None, fill=TEAL)
            page.insert_text((LEFT, y + size), stripped, fontsize=size, fontname="hebo", color=NAVY)
            y += size + 11
            index += 1
            continue
        indent = 12 if re.match(r"^\s*[-*]\s+", raw) or re.match(r"^\s*\d+\.\s+", raw) else 0
        wrapped = textwrap.wrap(stripped, width=103 if indent else 112, break_long_words=False, break_on_hyphens=False) or [""]
        height = 11 * len(wrapped) + 2
        if y + height > BOTTOM:
            page, y = new_page(document, "README | Pipeline overview")
        page.insert_textbox(fitz.Rect(LEFT + indent, y, RIGHT, y + height + 2), "\n".join(wrapped), fontsize=9.2, fontname="helv", color=INK, lineheight=1.25)
        y += height + 2
        index += 1


def add_figures(document: fitz.Document) -> None:
    page, y = new_page(document, "README | Results gallery")
    page.insert_text((LEFT, y), "3D results and comparisons", fontsize=20, fontname="hebo", color=NAVY)
    y += 30
    column_width = (RIGHT - LEFT - 18) / 2
    image_top = 150
    for figure_index in range(0, len(FIGURES), 2):
        row = FIGURES[figure_index:figure_index + 2]
        row_height = 0
        if y + image_top + 35 > BOTTOM:
            page, y = new_page(document, "README | Results gallery")
        for column, (filename, caption) in enumerate(row):
            path = FIGURE_DIR / filename
            if not path.exists():
                continue
            image = fitz.Pixmap(str(path))
            scale = min(column_width / image.width, image_top / image.height)
            width, height = image.width * scale, image.height * scale
            x = LEFT + column * (column_width + 18)
            page.insert_textbox(fitz.Rect(x, y, x + column_width, y + 22), caption, fontsize=9.5, fontname="hebo", color=INK)
            page.insert_image(fitz.Rect(x, y + 25, x + width, y + 25 + height), filename=str(path), keep_proportion=True)
            row_height = max(row_height, 25 + height)
        y += row_height + 22
def add_footer(document: fitz.Document) -> None:
    for number, page in enumerate(document, start=1):
        page.insert_text((LEFT, PAGE.height - 22), f"Tomofast-x 2.0 magnetic inversion pipeline  |  {number:02d}/{len(document):02d}", fontsize=7.5, fontname="helv", color=MUTED)


def main() -> None:
    document = fitz.open()
    add_body(document, README.read_text(encoding="utf-8").splitlines())
    add_figures(document)
    add_footer(document)
    document.set_metadata({"title": "Potential-field inversion pipeline", "author": "Tomofast-x 2.0 models"})
    document.save(OUTPUT, garbage=4, deflate=True)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes, {len(document)} pages)")


if __name__ == "__main__":
    main()
