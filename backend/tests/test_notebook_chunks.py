from app.teaching.chunks import clean_markdown, lecture_blocks, split_into_chunks


def test_notebook_splitter_keeps_urdu_and_english_sentence_boundaries() -> None:
    chunks = split_into_chunks("First idea. دوسری بات۔ Is this clear? جی ہاں؟")
    assert chunks == ["First idea.", "دوسری بات۔", "Is this clear?", "جی ہاں؟"]


def test_markdown_is_removed_and_blocks_are_structured() -> None:
    text = "**Definition:** A polynomial uses variables. Example: 3x + 5. Remember the exponent is whole."
    assert "**" not in clean_markdown(text)
    blocks = lecture_blocks(text)
    assert [block.type for block in blocks] == ["heading", "example", "note"]
    assert all(block.status == "pending" for block in blocks)
