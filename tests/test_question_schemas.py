"""Tests for question generation schemas."""

import pytest

from app.schemas.exam_content import GenerateQuestionsFromTopicRequest


def test_valid_request():
    """Test valid request creation."""
    request = GenerateQuestionsFromTopicRequest(
        topic="Phép cộng",
        grade_level="3",
        subject_code="T",
        questions_per_difficulty={"easy": 2, "medium": 1, "hard": 1},
        question_types=["multiple_choice", "fill_blank"],
    )
    assert request.topic == "Phép cộng"
    assert request.grade_level == "3"
    assert request.subject_code == "T"
    assert request.questions_per_difficulty["easy"] == 2


def test_invalid_grade_level():
    """Test invalid grade level rejection."""
    with pytest.raises(ValueError):
        GenerateQuestionsFromTopicRequest(
            topic="Test",
            grade_level="10",  # Invalid
            subject_code="T",
            questions_per_difficulty={"easy": 1},
            question_types=["multiple_choice"],
        )


def test_invalid_question_type():
    """Test invalid question type rejection."""
    with pytest.raises(ValueError):
        GenerateQuestionsFromTopicRequest(
            topic="Test",
            grade_level="3",
            subject_code="T",
            questions_per_difficulty={"easy": 1},
            question_types=["essay"],  # Invalid type
        )


def test_optional_additional_requirements():
    """Test that additional requirements are optional."""
    request = GenerateQuestionsFromTopicRequest(
        topic="Test Topic",
        grade_level="2",
        subject_code="TV",
        questions_per_difficulty={"easy": 3},
        question_types=["multiple_choice"],
    )
    assert request.additional_requirements is None


def test_provider_defaults():
    """Test that provider and model have defaults."""
    request = GenerateQuestionsFromTopicRequest(
        topic="Test",
        grade_level="1",
        subject_code="TA",
        questions_per_difficulty={"easy": 1},
        question_types=["true_false"],
    )
    assert request.provider == "google"
    assert request.model == "gemini-2.5-flash-lite"
