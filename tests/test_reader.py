from __future__ import annotations

import numpy as np

from ocr_sistemi import reader as reader_module
from ocr_sistemi.reader import TextBlock, extract_text, full_text

from conftest import ENGLISH_TEXT, INVOICE_LINE_1, INVOICE_LINE_2, TURKISH_TEXT, load


def test_extract_text_reads_english_sentence(english_line):
    blocks = extract_text(load(english_line), languages=("en",))

    assert len(blocks) == 1
    assert blocks[0].text == ENGLISH_TEXT


def test_extract_text_reads_turkish_sentence(turkish_line):
    blocks = extract_text(load(turkish_line), languages=("en", "tr"))

    assert len(blocks) == 1
    assert blocks[0].text == TURKISH_TEXT
    assert blocks[0].confidence > 0.8


def test_extract_text_finds_both_separate_regions(two_region_image):
    blocks = extract_text(load(two_region_image), languages=("en",))

    assert {b.text for b in blocks} == {INVOICE_LINE_1, INVOICE_LINE_2}


def test_extract_text_returns_empty_list_for_blank_image(blank_image):
    blocks = extract_text(load(blank_image), languages=("en",))

    assert blocks == []


def test_extract_text_box_has_four_corner_points(english_line):
    blocks = extract_text(load(english_line), languages=("en",))

    assert len(blocks[0].box) == 4
    assert all(len(point) == 2 for point in blocks[0].box)


def test_full_text_joins_blocks_with_newline():
    blocks = [
        TextBlock(text="birinci satır", confidence=0.9, box=((0, 0), (1, 0), (1, 1), (0, 1))),
        TextBlock(text="ikinci satır", confidence=0.8, box=((0, 2), (1, 2), (1, 3), (0, 3))),
    ]

    assert full_text(blocks) == "birinci satır\nikinci satır"


def test_full_text_of_empty_list_is_empty_string():
    assert full_text([]) == ""


def test_extract_text_filters_out_results_below_confidence_threshold(monkeypatch):
    class FakeReader:
        def readtext(self, image):
            return [
                (((0, 0), (10, 0), (10, 10), (0, 10)), "yüksek güven", 0.95),
                (((0, 20), (10, 20), (10, 30), (0, 30)), "düşük güven", 0.10),
            ]

    monkeypatch.setattr(reader_module, "_reader", lambda languages: FakeReader())

    blocks = extract_text(np.zeros((40, 40, 3), dtype=np.uint8), confidence_threshold=0.5)

    assert [b.text for b in blocks] == ["yüksek güven"]


def test_extract_text_lower_threshold_keeps_both(monkeypatch):
    class FakeReader:
        def readtext(self, image):
            return [
                (((0, 0), (10, 0), (10, 10), (0, 10)), "yüksek güven", 0.95),
                (((0, 20), (10, 20), (10, 30), (0, 30)), "düşük güven", 0.10),
            ]

    monkeypatch.setattr(reader_module, "_reader", lambda languages: FakeReader())

    blocks = extract_text(np.zeros((40, 40, 3), dtype=np.uint8), confidence_threshold=0.05)

    assert [b.text for b in blocks] == ["yüksek güven", "düşük güven"]
