from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from .reader import DEFAULT_CONFIDENCE, DEFAULT_LANGUAGES, extract_text, full_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocr-sistemi",
        description="Bir görüntüdeki tüm metni bulur ve okur — genel amaçlı OCR (İngilizce + Türkçe).",
    )
    parser.add_argument("image", help="metin aranacak görüntü dosyası")
    parser.add_argument(
        "--lang", default=",".join(DEFAULT_LANGUAGES),
        help=f"virgülle ayrılmış dil kodları (varsayılan: {','.join(DEFAULT_LANGUAGES)})",
    )
    parser.add_argument(
        "--confidence", type=float, default=DEFAULT_CONFIDENCE,
        help=f"minimum güven eşiği, 0-1 arası (varsayılan: {DEFAULT_CONFIDENCE})",
    )
    parser.add_argument(
        "--full-text", action="store_true",
        help="yalnızca birleştirilmiş düz metni yazdır (konum/güven bilgisi olmadan)",
    )
    return parser


def run(args) -> int:
    path = Path(args.image)
    if not path.exists():
        print(f"Hata: '{args.image}' bulunamadı.", file=sys.stderr)
        return 1

    image = cv2.imread(str(path))
    if image is None:
        print(f"Hata: '{args.image}' okunabilir bir görüntü formatında değil.", file=sys.stderr)
        return 1

    languages = tuple(lang.strip() for lang in args.lang.split(",") if lang.strip())
    blocks = extract_text(image, languages=languages, confidence_threshold=args.confidence)

    if not blocks:
        print("Görüntüde metin bulunamadı.")
        return 0

    if args.full_text:
        print(full_text(blocks))
        return 0

    for i, b in enumerate(blocks, 1):
        print(f'{i}. "{b.text}" (güven: {b.confidence:.2f})')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
