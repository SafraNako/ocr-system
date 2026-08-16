from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

DEFAULT_LANGUAGES = ("en", "tr")
DEFAULT_CONFIDENCE = 0.3


@dataclass(frozen=True)
class TextBlock:
    text: str
    confidence: float
    box: tuple[tuple[int, int], ...]  # 4 corner points, clockwise from top-left


@lru_cache(maxsize=4)
def _reader(languages: tuple[str, ...]):
    # Imported lazily and cached per language combination: constructing a
    # Reader loads model weights, which would otherwise happen on module
    # import even for callers that only want the dataclass/CLI parsing.
    import easyocr

    return easyocr.Reader(list(languages), gpu=False, verbose=False)


def extract_text(
    image: np.ndarray,
    *,
    languages: tuple[str, ...] = DEFAULT_LANGUAGES,
    confidence_threshold: float = DEFAULT_CONFIDENCE,
) -> list[TextBlock]:
    reader = _reader(tuple(languages))
    raw_results = reader.readtext(image)  # [(box, text, confidence), ...]

    blocks = []
    for box, text, confidence in raw_results:
        if confidence < confidence_threshold:
            continue
        points = tuple((int(x), int(y)) for x, y in box)
        blocks.append(TextBlock(text=text, confidence=float(confidence), box=points))
    return blocks


def full_text(blocks: list[TextBlock]) -> str:
    return "\n".join(b.text for b in blocks)
