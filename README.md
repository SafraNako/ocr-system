# ocr-system

[![Tests](https://github.com/SafraNako/ocr-system/actions/workflows/tests.yml/badge.svg)](https://github.com/SafraNako/ocr-system/actions/workflows/tests.yml)

General-purpose text extraction from images — English and Turkish, any
number of text regions, no assumptions about layout or format. Built on
EasyOCR, same engine as
[plate-recognition](https://github.com/SafraNako/plate-recognition), but a
different job: that one looks for exactly one plate-shaped rectangle and
validates it against a fixed format; this one reads whatever text is
anywhere in the image and makes no claims about what it should say.

## Usage

```bash
ocr-sistemi belge.jpg
```

```
1. "Fatura No: 2026-00142" (güven: 0.94)
2. "Toplam Tutar: 1.250,00 TL" (güven: 0.91)
3. "Teşekkür ederiz" (güven: 0.88)
```

```bash
# Just the plain text, no positions or confidence — good for piping
ocr-sistemi belge.jpg --full-text

# Only English (faster — skips loading the Turkish model)
ocr-sistemi belge.jpg --lang en

# Lower the confidence bar to catch more (and noisier) text
ocr-sistemi belge.jpg --confidence 0.1
```

## Setup

```bash
git clone https://github.com/SafraNako/ocr-system.git
cd ocr-system
pip install -e .
```

First run downloads EasyOCR's detection model plus a recognition model
per requested language, cached to `~/.EasyOCR/`.

## Development

```bash
pip install -e ".[dev]"
pytest -v
```

Test images are generated at test time with `Pillow` — real rendered
text at known positions, not real-world documents. Tests run the real
EasyOCR engine against them (no mocking) and check both recognition
accuracy on rendered text and pipeline behavior (confidence filtering,
language selection, empty-result handling).

## License

MIT — see [LICENSE](LICENSE).
