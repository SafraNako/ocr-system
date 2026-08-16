from __future__ import annotations

from ocr_sistemi import cli
from ocr_sistemi.reader import TextBlock

from conftest import ENGLISH_TEXT


def _parse(argv):
    return cli.build_parser().parse_args(argv)


def test_build_parser_defaults():
    args = _parse(["resim.jpg"])

    assert args.image == "resim.jpg"
    assert args.lang == "en,tr"
    assert args.full_text is False


def test_run_reports_missing_file(capsys):
    rc = cli.run(_parse(["olmayan_dosya.jpg"]))

    assert rc == 1
    assert "bulunamadı" in capsys.readouterr().err


def test_run_reports_unreadable_image(tmp_path, capsys):
    bad_file = tmp_path / "bozuk.jpg"
    bad_file.write_text("bu bir görüntü değil")

    rc = cli.run(_parse([str(bad_file)]))

    assert rc == 1
    assert "okunabilir bir görüntü formatında değil" in capsys.readouterr().err


def test_run_reports_no_text_found(monkeypatch, blank_image, capsys):
    monkeypatch.setattr(cli, "extract_text", lambda image, *, languages, confidence_threshold: [])

    rc = cli.run(_parse([str(blank_image)]))

    out = capsys.readouterr().out
    assert rc == 0
    assert "bulunamadı" in out


def test_run_prints_numbered_blocks_with_confidence(monkeypatch, english_line, capsys):
    blocks = [
        TextBlock(text="birinci", confidence=0.91, box=((0, 0), (1, 0), (1, 1), (0, 1))),
        TextBlock(text="ikinci", confidence=0.77, box=((0, 2), (1, 2), (1, 3), (0, 3))),
    ]
    monkeypatch.setattr(cli, "extract_text", lambda image, *, languages, confidence_threshold: blocks)

    rc = cli.run(_parse([str(english_line)]))

    out = capsys.readouterr().out
    assert rc == 0
    assert '1. "birinci" (güven: 0.91)' in out
    assert '2. "ikinci" (güven: 0.77)' in out


def test_run_full_text_flag_prints_joined_text_only(monkeypatch, english_line, capsys):
    blocks = [
        TextBlock(text="birinci satır", confidence=0.9, box=((0, 0), (1, 0), (1, 1), (0, 1))),
        TextBlock(text="ikinci satır", confidence=0.8, box=((0, 2), (1, 2), (1, 3), (0, 3))),
    ]
    monkeypatch.setattr(cli, "extract_text", lambda image, *, languages, confidence_threshold: blocks)

    rc = cli.run(_parse([str(english_line), "--full-text"]))

    out = capsys.readouterr().out
    assert rc == 0
    assert out == "birinci satır\nikinci satır\n"
    assert "güven" not in out


def test_run_passes_lang_and_confidence_through_to_extract_text(monkeypatch, english_line):
    captured = {}

    def fake_extract_text(image, *, languages, confidence_threshold):
        captured["languages"] = languages
        captured["confidence_threshold"] = confidence_threshold
        return []

    monkeypatch.setattr(cli, "extract_text", fake_extract_text)

    cli.run(_parse([str(english_line), "--lang", "en", "--confidence", "0.6"]))

    assert captured["languages"] == ("en",)
    assert captured["confidence_threshold"] == 0.6


def test_run_end_to_end_on_real_synthetic_image(english_line, capsys):
    rc = cli.run(_parse([str(english_line), "--lang", "en"]))

    out = capsys.readouterr().out
    assert rc == 0
    assert ENGLISH_TEXT in out
