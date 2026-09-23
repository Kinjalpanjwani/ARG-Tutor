import pytest

from app.speech.urdu_loanwords import (
    LOANWORD_TRANSLITERATIONS,
    transliterate_loanwords_for_urdu,
)


def test_replaces_known_loanword_with_urdu_script():
    assert transliterate_loanwords_for_urdu("computer system") == "کمپیوٹر system"


def test_title_case_and_uppercase_matched():
    assert transliterate_loanwords_for_urdu("Photosynthesis process") == "فوٹوسنتھیسس process"
    assert transliterate_loanwords_for_urdu("COMPUTER") == "کمپیوٹر"


def test_only_whole_words_are_replaced():
    # Prefix/suffix overlaps must not be touched.
    assert transliterate_loanwords_for_urdu("supercomputer computerized photosynthesis") == (
        "supercomputer computerized فوٹوسنتھیسس"
    )


def test_longest_term_wins_over_prefix():
    assert transliterate_loanwords_for_urdu("carbon dioxide") == "کاربن ڈائی آکسائیڈ"
    assert transliterate_loanwords_for_urdu("carbon") == "کاربن"


def test_mixed_roman_urdu_sentence_replaces_loanwords_only():
    sample = (
        "Photosynthesis wo process hai jismein plants glucose aur oxygen banate hain."
    )
    result = transliterate_loanwords_for_urdu(sample)
    assert "فوٹوسنتھیسس" in result
    assert "گلوکوز" in result
    assert "آکسیجن" in result
    assert "process" in result
    assert "plants" in result
    assert "aur" in result


def test_urdu_script_and_punctuation_untouched():
    text = "کیا آپ photosynthesis سمجھ رہے ہیں؟ (energy)"
    result = transliterate_loanwords_for_urdu(text)
    assert result == "کیا آپ فوٹوسنتھیسس سمجھ رہے ہیں؟ (انرجی)"


def test_empty_and_no_match_are_noops():
    assert transliterate_loanwords_for_urdu("") == ""
    assert transliterate_loanwords_for_urdu("کمپیوٹر") == "کمپیوٹر"
    assert transliterate_loanwords_for_urdu("no match here") == "no match here"


def test_replacement_is_idempotent():
    once = transliterate_loanwords_for_urdu("computer aur photosynthesis")
    twice = transliterate_loanwords_for_urdu(once)
    assert once == twice


def test_unknown_words_are_preserved():
    assert transliterate_loanwords_for_urdu("photosynthesis extra-unknownword") == (
        "فوٹوسنتھیسس extra-unknownword"
    )


def test_lookup_is_extensible_dict():
    assert "computer" in LOANWORD_TRANSLITERATIONS
    assert LOANWORD_TRANSLITERATIONS["computer"] == "کمپیوٹر"