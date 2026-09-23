from app.core.constants import (
    AttentionState,
    MoodState,
    VISION_CHECKIN_COOLDOWN_STEPS,
    VISION_NO_FACE_PAUSE_STREAK,
)
from app.language.routing import roman_urdu_instruction


def tone_directive_for_state(attention: str | None, mood: str | None) -> str | None:
    """Turns a smoothed (attention, mood) reading into a one-line tone directive for the LLM.
    Ported directly from notebook Cell 14c.
    """
    directive_parts: list[str] = []

    if attention == AttentionState.DISTRACTED:
        directive_parts.append(
            "The student seems to be looking away -- keep this next bit short, concrete, and re-engaging."
        )
    elif attention == AttentionState.DROWSY:
        directive_parts.append(
            "The student looks sleepy -- bring a bit more energy, maybe a quick question to re-hook them."
        )

    if mood == MoodState.HAPPY:
        directive_parts.append(
            "The student looks happy and engaged -- praise their progress and keep the energy up."
        )
    elif mood == MoodState.CONFUSED:
        directive_parts.append(
            "The student looks confused (furrowed brow) -- slow down and try a simpler example. "
            "Acknowledge it might be getting to be too much and ask something like: "
            "'I see this might be getting to be too much - would you like me to repeat that or take it slower?'"
        )
    elif mood == MoodState.STRESSED:
        directive_parts.append(
            "The student looks tense or frustrated -- be extra gentle and patient; slow down and simplify."
        )
    elif mood == MoodState.SAD:
        directive_parts.append(
            "The student looks a little down -- be warm and encouraging, and bring a touch more energy."
        )

    return " ".join(directive_parts).strip() or None


def build_checkin_prompt(directive: str, language: str = "English") -> str:
    """Creates prompt for generating one short, in-character check-in question."""
    base = (
        f"{directive} Based on this, ask the student ONE short, warm, in-character "
        "check-in question that they can actually answer -- offer a specific concrete "
        "fix (a simpler example, a slower pace, trying it a different way) and ask if "
        "they want that. Do not restate the lecture content. Just the one spoken "
        "question, no quotes."
    )
    if language in ("Urdu", "Roman Urdu"):
        base += (
            " Ask the question in Urdu (written Urdu script). Keep technical terms "
            "and key English vocabulary in English where that is the natural "
            "terminology."
        )
    return base


_ENGLISH_REPLY_RULE = (
    "Always reply in English, no matter what language or script the student used "
    "-- do not switch languages or mirror theirs."
)


def build_checkin_followup_prompt(
    question: str, directive: str, reply: str | None, last_chunk: str, language: str = "English"
) -> str:
    """Creates prompt for delivering the adapted explanation after a check-in reply."""
    if reply:
        prompt = (
            f'You just asked the student: "{question}" -- because {directive.strip()} '
            f'They replied: "{reply}". '
            f'Here is the point you were just teaching, to adapt: "{last_chunk}". '
            'Now: briefly acknowledge their reply, then actually deliver the adjusted '
            'version of that point (a simpler example if they wanted one, a slower or '
            "gentler explanation if that's what they wanted, or just move on warmly if "
            "they said they're fine) -- don't just say you'll do it, do it. End with one "
            'short natural sentence transitioning back into the lecture (e.g. moving on '
            'to the next point). This should read as ONE flowing spoken message, not '
            f'separate parts. {_ENGLISH_REPLY_RULE}'
        )
    else:
        prompt = (
            f'You just asked the student: "{question}" -- because {directive.strip()} '
            'They did not reply in time. '
            f'Here is the point you were just teaching: "{last_chunk}". '
            'Make the reasonable call yourself: gently deliver a simpler/slower/gentler '
            'version of that point without waiting further, then end with one short '
            'natural sentence transitioning back into the lecture. ONE flowing spoken '
            f'message. {_ENGLISH_REPLY_RULE}'
        )
    if language in ("Urdu", "Roman Urdu"):
        prompt = prompt.replace(
            _ENGLISH_REPLY_RULE, roman_urdu_instruction("Urdu")
        )
    return prompt


def build_content_checkin_followup_prompt(
    question: str, reply: str, last_chunk: str, language: str = "English"
) -> str:
    """Prompt for following up on a content-driven (scheduled) check-in: affirm a right
    answer, correct a wrong one with reasoning, then transition back into the lesson."""
    prompt = (
        f'You paused mid-lesson to ask the student: "{question}". They replied: '
        f'"{reply}". Here is the point you were just teaching: "{last_chunk}". '
        "Now: warmly acknowledge their reply. If their answer is right, affirm it "
        "clearly; if it is wrong or partially wrong, gently correct it and show the "
        "reasoning in one line. Then continue naturally from where you stopped, keeping "
        "this to one or two spoken sentences that flow as one message, and do not start "
        f"a new topic. {_ENGLISH_REPLY_RULE}"
    )
    if language in ("Urdu", "Roman Urdu"):
        prompt = prompt.replace(
            _ENGLISH_REPLY_RULE, roman_urdu_instruction("Urdu")
        )
    return prompt

