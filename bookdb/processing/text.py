
from __future__ import annotations

import re


def is_non_informative(text: str) -> bool:
    """Return True if the text carries no useful information.

    Catches: single characters, punctuation-only, digits-only, and
    repeated-character strings (e.g. "aaaa").

    Args:
        text: Raw review or free-text string.

    Returns:
        True if the text should be considered non-informative.
    """
    if not text:
        return False
    text = text.strip()
    if not text:
        return False
    if len(text) == 1:
        return True
    if re.fullmatch(r"[\W_]+", text):
        return True
    if re.fullmatch(r"[\d\s]+", text):
        return True
    if re.fullmatch(r"(.)\1+", text):
        return True
    return False


def is_short_low_variety(text: str) -> bool:
    """Return True if the text is too short and uses too few distinct characters.

    Threshold: length <= 5 and <= 2 unique non-space characters.

    Args:
        text: Raw review or free-text string.

    Returns:
        True if the text is short and low-variety.
    """
    if not text:
        return False
    text = text.strip()
    if not text:
        return False
    return len(text) <= 5 and len(set(text.replace(" ", ""))) <= 2
