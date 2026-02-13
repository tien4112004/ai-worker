"""Test FILL_IN_BLANK schema validation."""

import pytest
from pydantic import ValidationError

from app.schemas.exam_content import Question


def test_fill_in_blank_with_string_data():
    """Test that FILL_IN_BLANK accepts FillInBlankStringData."""
    question_data = {
        "type": "FILL_IN_BLANK",
        "difficulty": "KNOWLEDGE",
        "title": "Complete the sentence",
        "grade": "3",
        "chapter": "Test Topic",
        "subject": "TV",
        "data": {
            "type": "FILL_IN_BLANK",
            "data": "The capital of Vietnam is {{Hà Nội|Ha Noi|Hanoi}}.",
        },
        "explanation": "Hanoi is the capital city.",
        "point": 1.0,
    }

    question = Question(**question_data)
    assert question.type == "FILL_IN_BLANK"
    assert hasattr(question.data, "data")
    assert isinstance(question.data.data, str)
    assert "{{" in question.data.data
    assert "}}" in question.data.data


def test_fill_in_blank_with_multiple_blanks():
    """Test FILL_IN_BLANK with multiple blanks in one string."""
    question_data = {
        "type": "FILL_IN_BLANK",
        "difficulty": "COMPREHENSION",
        "title": "Fill in the blanks",
        "grade": "4",
        "chapter": "Science",
        "subject": "T",
        "data": {
            "type": "FILL_IN_BLANK",
            "data": "Plants need {{water|nước}} and {{sunlight|ánh sáng}} to grow.",
        },
        "explanation": "Basic photosynthesis needs.",
        "point": 2.0,
    }

    question = Question(**question_data)
    assert question.type == "FILL_IN_BLANK"
    assert hasattr(question.data, "data")
    # Should have 2 blank placeholders
    assert question.data.data.count("{{") == 2
    assert question.data.data.count("}}") == 2


def test_multiple_choice_with_object_data():
    """Test that MULTIPLE_CHOICE still uses object data."""
    question_data = {
        "type": "MULTIPLE_CHOICE",
        "difficulty": "KNOWLEDGE",
        "title": "What is 2+2?",
        "grade": "1",
        "chapter": "Math",
        "subject": "T",
        "data": {
            "type": "MULTIPLE_CHOICE",
            "options": [
                {"text": "3", "isCorrect": False},
                {"text": "4", "isCorrect": True},
                {"text": "5", "isCorrect": False},
                {"text": "6", "isCorrect": False},
            ],
            "shuffleOptions": True,
        },
        "explanation": "2+2=4",
        "point": 1.0,
    }

    question = Question(**question_data)
    assert question.type == "MULTIPLE_CHOICE"
    assert hasattr(question.data, "options")


def test_fill_in_blank_with_case_sensitive():
    """Test that FillInBlankData supports caseSensitive field."""
    question_data = {
        "type": "FILL_IN_BLANK",
        "difficulty": "KNOWLEDGE",
        "title": "Test",
        "grade": "3",
        "chapter": "Test",
        "subject": "TV",
        "data": {
            "type": "FILL_IN_BLANK",
            "data": "Hello {{world|World}}",
            "caseSensitive": True,
        },
        "point": 1.0,
    }

    question = Question(**question_data)
    assert question.type == "FILL_IN_BLANK"
    assert hasattr(question.data, "data")
    assert hasattr(question.data, "caseSensitive")
    assert question.data.caseSensitive == True
    assert "{{" in question.data.data


def test_invalid_question_type():
    """Test that invalid question type is rejected."""
    question_data = {
        "type": "INVALID_TYPE",
        "difficulty": "KNOWLEDGE",
        "title": "Test",
        "grade": "3",
        "chapter": "Test",
        "subject": "TV",
        "data": {"type": "FILL_IN_BLANK", "data": "test"},
        "point": 1.0,
    }

    with pytest.raises(ValidationError):
        Question(**question_data)


def test_missing_required_fields():
    """Test that missing required fields are rejected."""
    question_data = {
        "type": "FILL_IN_BLANK",
        "difficulty": "KNOWLEDGE",
        # Missing: title, grade, chapter, subject, data
    }

    with pytest.raises(ValidationError):
        Question(**question_data)
