"""Subject-specific prompt routing utilities.

This module provides functionality to select and combine subject-specific
system prompts based on subject codes and grades.
"""

from typing import Optional

# Mapping: subject_code -> registry key
# T = Toán (Math), TV = Tiếng Việt (Vietnamese/Literature), TA = Tiếng Anh (English)
SUBJECT_PROMPT_MAP = {
    "T": "subject.math",
    "TV": "subject.literature",
    "TA": "subject.english",
}


def get_subject_prompt_key(subject_code: Optional[str]) -> Optional[str]:
    """Get the prompt registry key for a subject code.

    Args:
        subject_code: The subject code (e.g., 'T', 'TV', 'TA')

    Returns:
        The registry key for the subject prompt, or None if not found
    """
    if not subject_code:
        return None
    return SUBJECT_PROMPT_MAP.get(subject_code)


def combine_system_prompts(
    base_prompt: str, subject_prompt: Optional[str]
) -> str:
    """Combine base and subject prompts with proper formatting.

    Args:
        base_prompt: The base system prompt
        subject_prompt: The subject-specific system prompt (optional)

    Returns:
        Combined prompt string
    """
    if not subject_prompt:
        return base_prompt

    # Combine with clear separation
    return f"""{base_prompt}

---

# SUBJECT-SPECIFIC LEARNING OBJECTIVES

{subject_prompt}"""
