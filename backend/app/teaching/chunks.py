import json
import re
from uuid import uuid4

from app.schemas.lesson import BlockType, WhiteboardBlock


MARKDOWN_TOKENS = re.compile(r"(?:\*\*|__|`|^\s{0,3}#{1,6}\s*|^\s*[-*+]\s+)", re.MULTILINE)


def clean_markdown(text: str) -> str:
    return re.sub(r"\s+", " ", MARKDOWN_TOKENS.sub("", text)).strip()


# Whiteboard-friendly upper bound for a single prose block. Blocks above this are
# split into several smaller blocks so the board never shows one giant paragraph.
BLOCK_SPLIT_LIMIT = 240


def _split_prose_sentences(text: str) -> list[str]:
    """Sentence-aware split shared by the chunker and the over-length block splitter."""
    clean = clean_markdown(text)
    parts = re.split(r"([۔؟?!.]+)", clean)
    sentences: list[str] = []
    for index in range(0, len(parts) - 1, 2):
        sentence = (parts[index] + parts[index + 1]).strip()
        if sentence:
            sentences.append(sentence)
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())
    return sentences


def split_prose_for_whiteboard(text: str, limit: int = BLOCK_SPLIT_LIMIT) -> list[str]:
    """Split over-length prose into whiteboard-sized pieces on line and sentence
    boundaries. Numbered bullet lines (from interrupt answers) are preserved as units."""
    text = text.strip()
    if not text or len(text) <= limit:
        return [text] if text else []
    pieces: list[str] = []
    for line in re.split(r"\n+", text):
        line = line.strip()
        if not line:
            continue
        if len(line) <= limit:
            pieces.append(line)
            continue
        current = ""
        for sentence in re.split(r"(?<=[.?!]) (?=[A-Z0-9آ-ی])", line):
            candidate = f"{current} {sentence}".strip()
            if len(candidate) <= limit:
                current = candidate
                continue
            if current:
                pieces.append(current)
                current = ""
            while len(sentence) > limit:
                cut = sentence[:limit].rstrip()
                boundary = cut.rfind(" ")
                if boundary > limit * 0.6:
                    cut = cut[:boundary]
                pieces.append(cut)
                sentence = sentence[len(cut):].lstrip()
            current = sentence
        if current:
            pieces.append(current)
    return [piece for piece in pieces if piece]


def _collapse_json_shell(text: str) -> str:
    """Unwrap a stray JSON payload an LLM sometimes returns in place of spoken prose
    (e.g. a full lecture JSON on a check-in question) into flat readable text."""
    stripped = text.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return text
    try:
        payload = json.loads(stripped)
    except Exception:
        return text

    pieces: list[str] = []

    def flatten(item):
        if isinstance(item, dict):
            for key in ("lesson_title", "heading", "question", "content", "text"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    pieces.append(value.strip())
            for value in item.values():
                if isinstance(value, (dict, list)):
                    flatten(value)
        elif isinstance(item, list):
            for entry in item:
                flatten(entry)

    flatten(payload)
    deduplicated = " ".join(dict.fromkeys(pieces)).strip()
    return deduplicated or text


def clean_spoken_text(text: str) -> str:
    """Prepare an LLM 'spoken' answer for a whiteboard block: strip markdown,
    re-separate numbered bullet lines into their own lines, and unwrap any stray
    JSON payload the model returned in place of prose."""
    cleaned = clean_markdown(text)
    cleaned = re.sub(r"\s+(?=\d{1,2}\.\s)", "\n", cleaned)
    unwrapped = _collapse_json_shell(cleaned).strip()
    return unwrapped or text.strip()


def split_into_chunks(text: str, sentences_per_chunk: int = 1) -> list[str]:
    """Port of the fixed notebook's punctuation-aware sentence splitter."""
    clean = clean_markdown(text)
    # The notebook pattern is preserved and extended with ASCII "?" so English
    # questions are not accidentally joined to the following Urdu sentence.
    parts = re.split(r"([۔؟?!.]+)", clean)
    sentences: list[str] = []
    for index in range(0, len(parts) - 1, 2):
        sentence = (parts[index] + parts[index + 1]).strip()
        if sentence:
            sentences.append(sentence)
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())
    return [
        " ".join(sentences[index:index + sentences_per_chunk]).strip()
        for index in range(0, len(sentences), sentences_per_chunk)
        if sentences[index:index + sentences_per_chunk]
    ]


def infer_block_type(text: str, index: int) -> BlockType:
    lowered = text.casefold()
    if index == 0:
        return "heading"
    if any(token in lowered for token in ("for example", "example:", "consider ")):
        return "example"
    if any(token in lowered for token in ("remember", "important", "note that")):
        return "note"
    if re.search(r"(?:=|[a-z]\^\d|\b[a-z]\s*[+\-*/]\s*\d)", text, re.I):
        return "equation"
    return "text"


def lecture_blocks(lecture_text: str) -> list[WhiteboardBlock]:
    return [
        WhiteboardBlock(
            id=str(uuid4()), type=infer_block_type(chunk, index), text=chunk,
            latex=chunk if infer_block_type(chunk, index) == "equation" else None,
        )
        for index, chunk in enumerate(split_into_chunks(lecture_text))
    ]
