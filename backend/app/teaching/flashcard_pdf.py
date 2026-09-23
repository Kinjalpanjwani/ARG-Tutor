from __future__ import annotations

import io
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from app.schemas.lesson import Flashcard

_FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"

_BOARD = HexColor("#163f3a")
_PANEL = HexColor("#14332f")
_PANEL_EDGE = HexColor("#3f7a6f")
_CHALK = HexColor("#f5f1da")
_CHALK_SOFT = HexColor("#d7e5dc")
_LIME = HexColor("#c9ef80")
_TEAL = HexColor("#9ac2b8")
_FRAME_INNER = HexColor("#1e4a42")

_PAGE_W, _PAGE_H = A4
_MARGIN = 12 * mm
_PAD_X = 5 * mm
_PAD_Y = 4 * mm
_GAP_X = 7 * mm
_GAP_Y = 5 * mm
_COLS = 2
_ROWS = 3
_CAPACITY = _COLS * _ROWS

_FONTS = {
    "DMMono": "DMMono-Regular.ttf",
    "DMMono-Medium": "DMMono-Medium.ttf",
    "Literata": "Literata-500.ttf",
    "Literata-600": "Literata-600.ttf",
    "PublicSans": "PublicSans-Regular.ttf",
    "PublicSans-SemiBold": "PublicSans-SemiBold.ttf",
}
_FONTS_REGISTERED = False


def _register_fonts() -> None:
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    for name, file in _FONTS.items():
        path = _FONT_DIR / file
        if path.exists():
            pdfmetrics.registerFont(TTFont(name, str(path)))
    _FONTS_REGISTERED = True


def _wrap(text: str, font: str, size: float, max_width: float) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        if pdfmetrics.stringWidth(candidate, font, size) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_label(c: canvas.Canvas, text: str, x: float, y: float, color, right: bool = False) -> None:
    step = 0.25
    width = sum(pdfmetrics.stringWidth(char, "DMMono", 5.5) + step for char in text) - step
    cursor = x - width if right else x
    c.setFillColor(color)
    c.setFont("DMMono", 5.5)
    for char in text:
        c.drawString(cursor, y, char)
        cursor += pdfmetrics.stringWidth(char, "DMMono", 5.5) + step


def _draw_panel(c: canvas.Canvas, card: Flashcard, x: float, y: float, w: float, h: float, language: str | None) -> None:
    c.setFillColor(_PANEL)
    c.setStrokeColor(_PANEL_EDGE)
    c.setLineWidth(0.9)
    c.roundRect(x, y, w, h, 9, stroke=1, fill=1)

    inner_w = w - 2 * _PAD_X
    top_lbl = y + h - _PAD_Y - 2
    _draw_label(c, "TERM", x + _PAD_X, top_lbl, _LIME)
    if language:
        _draw_label(c, language, x + w - _PAD_X, top_lbl, _TEAL, right=True)

    front_size = 12.5
    front_lines = _wrap(card.front, "Literata", front_size, inner_w)
    if len(front_lines) > 3:
        front_size = 11
        front_lines = _wrap(card.front, "Literata", front_size, inner_w)
    line_h = front_size * 1.4
    baseline = top_lbl - 8
    c.setFillColor(_CHALK)
    c.setFont("Literata", front_size)
    for line in front_lines:
        c.drawString(x + _PAD_X, baseline, line)
        baseline -= line_h
    last_line_y = baseline + line_h

    divider_y = last_line_y - 5
    c.setStrokeColor(_PANEL_EDGE)
    c.setLineWidth(0.5)
    c.line(x + _PAD_X, divider_y, x + w - _PAD_X, divider_y)

    answer_lbl_y = divider_y - 7
    _draw_label(c, "ANSWER", x + _PAD_X, answer_lbl_y, _TEAL)

    back_start = answer_lbl_y - 6
    floor = y + _PAD_Y + 7
    back_size = 8.8
    while back_size > 7.5:
        back_lines = _wrap(card.back, "PublicSans", back_size, inner_w)
        if back_start - len(back_lines) * back_size * 1.35 >= floor:
            break
        back_size -= 0.3
    else:
        back_lines = _wrap(card.back, "PublicSans", back_size, inner_w)
    c.setFillColor(_CHALK_SOFT)
    c.setFont("PublicSans", back_size)
    baseline = back_start
    for line in back_lines:
        c.drawString(x + _PAD_X, baseline, line)
        baseline -= back_size * 1.35


def render(cards: list[Flashcard], topic: str, language: str) -> bytes:
    _register_fonts()
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4, bottomup=1)
    c.setTitle(f"ARG Tutor Flashcards · {topic}")

    top = _PAGE_H - _MARGIN
    content_x = _MARGIN + 3 * mm
    content_w = _PAGE_W - 2 * content_x
    card_w = (content_w - (_COLS - 1) * _GAP_X) / _COLS
    header_bottom = top - 38 * mm
    footer_y = _MARGIN + 4 * mm
    usable_top = header_bottom
    usable_bottom = footer_y + 4 * mm
    card_h = (usable_top - usable_bottom - (_ROWS - 1) * _GAP_Y) / _ROWS

    pages = [cards[i : i + _CAPACITY] for i in range(0, len(cards), _CAPACITY)] or [[]]
    for page_no, batch in enumerate(pages, start=1):
        c.setFillColor(_BOARD)
        c.rect(0, 0, _PAGE_W, _PAGE_H, stroke=0, fill=1)
        c.setStrokeColor(_FRAME_INNER)
        c.setLineWidth(1.1)
        c.roundRect(_MARGIN, _MARGIN, _PAGE_W - 2 * _MARGIN, _PAGE_H - 2 * _MARGIN, 10, stroke=1, fill=0)

        c.setFillColor(_LIME)
        c.setFont("DMMono-Medium", 7.5)
        c.drawRightString(_PAGE_W - _MARGIN - 3 * mm, top - 7, "ARG TUTOR · FLASHCARDS")
        c.setFont("DMMono-Medium", 8)
        c.drawString(content_x, top - 7, "STUDY CARDS")
        c.setFillColor(_CHALK)
        c.setFont("Literata-600", 15)
        c.drawString(content_x, top - 24, topic)
        if language:
            c.setFillColor(_TEAL)
            c.setFont("DMMono", 7.5)
            c.drawRightString(_PAGE_W - _MARGIN - 3 * mm, top - 24, language)
        c.setStrokeColor(_PANEL_EDGE)
        c.setLineWidth(0.6)
        c.line(content_x, top - 32, _PAGE_W - content_x, top - 32)

        for index, card in enumerate(batch):
            col = index % _COLS
            row = index // _COLS
            x = content_x + col * (card_w + _GAP_X)
            y = usable_top - card_h - row * (card_h + _GAP_Y)
            _draw_panel(c, card, x, y, card_w, card_h, language)

        c.setFillColor(_TEAL)
        c.setFont("DMMono", 7)
        c.drawCentredString(_PAGE_W / 2, footer_y, f"· {page_no} ·")
        c.showPage()

    c.save()
    return buffer.getvalue()