import re
from uuid import uuid4

from app.schemas.lesson import BlockType, WhiteboardBlock


MARKDOWN_TOKENS = re.compile(r"(?:\*\*|__|`|^\s{0,3}#{1,6}\s*|^\s*[-*+]\s+)", re.MULTILINE)


def clean_markdown(text: str) -> str:
    return re.sub(r"\s+", " ", MARKDOWN_TOKENS.sub("", text)).strip()


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
