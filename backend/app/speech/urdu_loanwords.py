"""Pre-synthesis transliteration of English loan words for the Urdu TTS voice.

The Urdu tutor voice (e.g. ur-PK-AsadNeural) mispronounces English words that are
embedded in the spoken text (lesson chunks, hard-coded Urdu-script copy). We cannot
switch the voice per-span with SSML because edge-tts always XML-escapes the text
and wraps it in a fixed <voice> element, so the only reliable fix is to rewrite
known loan words to their Urdu-script phonetic spelling before synthesis. The Urdu
network voice then pronounces them with a native Urdu accent.

To extend the list: add entries here as ``"english word": "اردو املا"``. Matching is
whole-word, case-insensitive, and longest-match-first, so multi-word terms such as
"carbon dioxide" and single-word prefixes like "photo" both work.
"""

import re

LOANWORD_TRANSLITERATIONS: dict[str, str] = {
    # Science / biology
    "photosynthesis": "فوٹوسنتھیسس",
    "photo": "فوٹو",
    "chlorophyll": "کلوروفیل",
    "chloroplast": "کلوروپلاسٹ",
    "stomata": "سٹوماٹا",
    "glucose": "گلوکوز",
    "protein": "پروٹین",
    "carbohydrate": "کاربوہائیڈریٹ",
    "vitamin": "وٹامن",
    "molecule": "مالیکیول",
    "electron": "الیکٹران",
    "proton": "پروٹان",
    "enzyme": "انزائم",
    "hormone": "ہارمون",
    "bacteria": "بیکٹیریا",
    "virus": "وائرس",
    "cell": "سیل",
    "brain": "دماغ",
    "neuron": "نیوران",
    # Chemistry / physics
    "chemistry": "کیمسٹری",
    "physics": "فزکس",
    "oxygen": "آکسیجن",
    "hydrogen": "ہائیڈروجن",
    "carbon": "کاربن",
    "carbon dioxide": "کاربن ڈائی آکسائیڈ",
    "energy": "انرجی",
    "atom": "ایٹم",
    # Other subjects / everyday loans
    "biology": "بائیولوجی",
    "science": "سائنس",
    "computer": "کمپیوٹر",
    "nutrition": "نیوٹریشن",
}

_KEY_PATTERN = re.compile(
    r"(?<![A-Za-z])(?:"
    + "|".join(
        re.escape(key)
        for key in sorted(LOANWORD_TRANSLITERATIONS, key=len, reverse=True)
    )
    + r")(?![A-Za-z])",
    re.IGNORECASE,
)


def transliterate_loanwords_for_urdu(text: str) -> str:
    """Replace recognized English loan words with their Urdu-script spellings.

    Non-matching text (including Urdu script and any other language) is left
    untouched. The pass is idempotent because the replacements are Urdu-script
    words, which the pattern can never match again.
    """
    if not text:
        return text

    def _repl(match: re.Match) -> str:
        return LOANWORD_TRANSLITERATIONS[match.group(0).lower()]

    return _KEY_PATTERN.sub(_repl, text)