"""Subject-specific prompt routing utilities.

This module provides functionality to select and combine subject-specific
system prompts based on subject codes and grades.
"""

from typing import Optional

# Mapping: subject_code -> registry key base
# T = Toán (Math), TV = Tiếng Việt (Vietnamese/Literature), TA = Tiếng Anh (English)
SUBJECT_PROMPT_MAP = {
    "T": "math",
    "TV": "literature",
    "TA": "english",
}


def get_subject_grade_prompt_key(
    subject_code: Optional[str], grade: Optional[str]
) -> Optional[str]:
    """Get the prompt registry key for a subject-grade combination.

    Args:
        subject_code: The subject code (e.g., 'T', 'TV', 'TA')
        grade: The grade level (e.g., '1', '2', '3', '4', '5')

    Returns:
        The registry key for the subject-grade prompt, or None if not found
        Example: 'subject_grade.math.1' for subject='T', grade='1'
    """
    if not subject_code or not grade:
        return None

    subject_name = SUBJECT_PROMPT_MAP.get(subject_code)
    if not subject_name:
        return None

    # Build the registry key: subject_grade.{subject}.{grade}
    return f"subject_grade.{subject_name}.{grade}"


def get_subject_prompt_key(subject_code: Optional[str]) -> Optional[str]:
    """Get the prompt registry key for a subject code (legacy, for backward compatibility).

    Args:
        subject_code: The subject code (e.g., 'T', 'TV', 'TA')

    Returns:
        The registry key for the subject prompt, or None if not found
    """
    if not subject_code:
        return None

    subject_name = SUBJECT_PROMPT_MAP.get(subject_code)
    if not subject_name:
        return None

    return f"subject.{subject_name}"
