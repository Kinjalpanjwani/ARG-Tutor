from app.language.routing import voice_for


def test_voice_for_maps_english_to_english_voice():
    assert voice_for("English", english_voice="en-US-AriaNeural", urdu_voice="ur-PK-AsadNeural") == "en-US-AriaNeural"


def test_voice_for_maps_urdu_to_urdu_voice():
    assert voice_for("Urdu", english_voice="en-US-AriaNeural", urdu_voice="ur-PK-AsadNeural") == "ur-PK-AsadNeural"


def test_voice_for_collapses_roman_urdu_to_urdu_voice():
    assert voice_for("Roman Urdu", english_voice="en-US-AriaNeural", urdu_voice="ur-PK-AsadNeural") == "ur-PK-AsadNeural"


def test_voice_for_unknown_language_falls_back_to_english_voice():
    assert voice_for("Klingon", english_voice="en-US-AriaNeural", urdu_voice="ur-PK-AsadNeural") == "en-US-AriaNeural"