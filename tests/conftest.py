from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

# PIL's ImageFont.load_default() has no glyphs for Turkish-specific
# characters (ı, ş, ğ, ü) and silently renders them as the wrong shape —
# EasyOCR then reads garbage. DejaVu Sans has full coverage and is bundled
# here (not relied on as a system font) so this works identically on the
# dev Mac and in Linux CI.
FONT_PATH = Path(__file__).parent / "fixtures" / "DejaVuSans.ttf"

ENGLISH_TEXT = "The quick brown fox jumps"
TURKISH_TEXT = "Işığın hızı çok yüksektir"
INVOICE_LINE_1 = "Fatura No 2026-00142"
INVOICE_LINE_2 = "Toplam Tutar 1250 TL"


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size)


def _save(image: Image.Image, path: Path) -> Path:
    image.save(path)
    return path


def make_single_line_image(path: Path, text: str, *, font_size: int = 36) -> Path:
    image = Image.new("RGB", (700, 120), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((15, 30), text, fill="black", font=_font(font_size))
    return _save(image, path)


def make_two_region_image(path: Path) -> Path:
    """Two well-separated text blocks — checks that regions aren't merged
    or dropped, not just that OCR reads a single string correctly."""
    image = Image.new("RGB", (700, 400), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 40), INVOICE_LINE_1, fill="black", font=_font(32))
    draw.text((20, 280), INVOICE_LINE_2, fill="black", font=_font(32))
    return _save(image, path)


def make_blank_image(path: Path) -> Path:
    image = Image.new("RGB", (700, 400), color="white")
    return _save(image, path)


@pytest.fixture
def english_line(tmp_path) -> Path:
    return make_single_line_image(tmp_path / "en.png", ENGLISH_TEXT)


@pytest.fixture
def turkish_line(tmp_path) -> Path:
    return make_single_line_image(tmp_path / "tr.png", TURKISH_TEXT)


@pytest.fixture
def two_region_image(tmp_path) -> Path:
    return make_two_region_image(tmp_path / "iki_bolge.png")


@pytest.fixture
def blank_image(tmp_path) -> Path:
    return make_blank_image(tmp_path / "bos.png")


def load(path: Path) -> np.ndarray:
    """Load the same way the CLI does — through cv2.imread — so tests
    exercise the real BGR array shape the reader sees in production."""
    return cv2.imread(str(path))
