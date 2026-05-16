"""
Helpers for stopping generation at a natural story boundary.

Our training data concatenates stories without <|endoftext|> markers (see
prepare_data.py in Episode 05), so the model rarely samples EOS (token 0).
When that happens, we fall back to cutting before a repeated story opener.
"""

import re

# Second story in the same generation usually starts right after . ! or ?
_STORY_RESTART_RE = re.compile(
    r'(?<=[.!?])"?\s*'
    r'(?:Once upon a time\b|One sunny morning,|One day, there was\b|There was a little\b)',
    re.IGNORECASE,
)


def truncate_at_story_restart(text: str) -> tuple[str, bool]:
    """
    If generated text contains a second story opener, return text before it.

    Returns:
        (trimmed_text, did_truncate)
    """
    m = _STORY_RESTART_RE.search(text)
    if m:
        return text[: m.start()].rstrip(), True
    return text, False
