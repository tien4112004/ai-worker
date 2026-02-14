"""Metadata parser module to extract educational metadata from document filenames.

This module parses Vietnamese educational book filenames to extract:
- Grade level (1-5)
- Subject code (T, TA, TV)
- Subject name in Vietnamese

Expected filename format:
- Math: SGV_KNTT_Tx (e.g., SGV_KNTT_T1.pdf)
- English: SGV_KNTT_TAx (e.g., SGV_KNTT_TA3.pdf)
- Literature: SGV_KNTT_TVx_Ty (e.g., SGV_KNTT_TV5_T2.pdf)

Where x is the grade (1-5) and y is the version (ignored).
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional

# Subject code mapping
SUBJECT_MAPPING = {
    "T": "Toán",  # Math
    "TA": "Tiếng Anh",  # English
    "TV": "Tiếng Việt",  # Literature
}


def parse_educational_metadata(filename: str) -> Dict[str, Any]:
    """
    Parse educational metadata from a Vietnamese educational book filename.

    Args:
        filename: The filename (with or without extension)

    Returns:
        Dictionary containing:
        - grade: Integer (1-5) or None if not found
        - subject_code: String (T/TA/TV) or None if not found
        - subject_name: String (Vietnamese name) or None if not found
        - has_metadata: Boolean indicating if metadata was successfully extracted

    Examples:
        >>> parse_educational_metadata("SGV_KNTT_T1.pdf")
        {'grade': 1, 'subject_code': 'T', 'subject_name': 'Toán', 'has_metadata': True}

        >>> parse_educational_metadata("SGV_KNTT_TA3.pdf")
        {'grade': 3, 'subject_code': 'TA', 'subject_name': 'Tiếng Anh', 'has_metadata': True}

        >>> parse_educational_metadata("SGV_KNTT_TV5_T2.pdf")
        {'grade': 5, 'subject_code': 'TV', 'subject_name': 'Tiếng Việt', 'has_metadata': True}
    """
    # Remove extension and get base filename
    base_filename = Path(filename).stem

    # Initialize result
    result = {
        "grade": None,
        "subject_code": None,
        "subject_name": None,
        "has_metadata": False,
    }

    # Pattern to match the educational book filename format
    # Matches: SGV_KNTT_<SUBJECT><GRADE>[_T<VERSION>] or SGK_<SUBJECT><GRADE>[_<VERSION>]
    # Subject can be: T, TA, or TV
    # Grade is a single digit (1-5)
    # Optional version suffix for literature books (e.g., _T)
    pattern = r"SG[VK]_KNTT_(T[AV]?)(\d)(?:_T\d)?$"

    match = re.search(pattern, base_filename, re.IGNORECASE)

    if match:
        subject_code = match.group(1).upper()
        grade_str = match.group(2)

        print("subject_code", subject_code)
        print("grade_str", grade_str)

        try:
            grade = int(grade_str)

            # Validate grade is in range 1-5
            if 1 <= grade <= 5:
                # Validate subject code
                if subject_code in SUBJECT_MAPPING:
                    result["grade"] = grade
                    result["subject_code"] = subject_code
                    result["subject_name"] = SUBJECT_MAPPING[subject_code]
                    result["has_metadata"] = True
        except ValueError:
            pass

    return result


def extract_metadata_from_path(file_path: str) -> Dict[str, Any]:
    """
    Extract educational metadata from a file path.

    This is a convenience wrapper around parse_educational_metadata
    that accepts a full file path.

    Args:
        file_path: Full path to the file

    Returns:
        Dictionary with educational metadata
    """
    filename = Path(file_path).name
    return parse_educational_metadata(filename)


def validate_metadata(metadata: Dict[str, Any]) -> bool:
    """
    Validate that metadata contains required educational fields.

    Args:
        metadata: Metadata dictionary to validate

    Returns:
        True if metadata is valid, False otherwise
    """
    return (
        metadata.get("has_metadata", False) is True
        and metadata.get("grade") is not None
        and metadata.get("subject_code") is not None
        and metadata.get("subject_name") is not None
    )


def get_metadata_summary(metadata: Dict[str, Any]) -> str:
    """
    Get a human-readable summary of the metadata.

    Args:
        metadata: Metadata dictionary

    Returns:
        Formatted string describing the metadata
    """
    if not validate_metadata(metadata):
        return "No educational metadata found"

    return f"Grade {metadata['grade']} - {metadata['subject_name']} ({metadata['subject_code']})"
