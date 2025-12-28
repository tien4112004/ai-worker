"""Schemas for exam and question generation."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class MatrixItem(BaseModel):
    """Represents a single item in the exam matrix."""

    topic: str = Field(..., description="Topic or subtopic for questions")
    question_type: Literal[
        "multiple_choice", "true_false", "fill_blank", "long_answer", "matching"
    ] = Field(..., description="Type of question")
    count: int = Field(..., ge=1, description="Number of questions to generate")
    points_each: int = Field(..., ge=1, description="Points for each question")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ..., description="Difficulty level"
    )
    requires_context: bool = Field(
        default=False, description="Whether question requires a context/passage"
    )
    context_type: Optional[
        Literal["reading_passage", "image", "audio", "video"]
    ] = Field(None, description="Type of context if required")


class GenerateMatrixRequest(BaseModel):
    """Request to generate an exam matrix using AI."""

    topic: str = Field(..., description="Main topic of the exam")
    grade_level: Literal["K", "1", "2", "3", "4", "5"] = Field(
        ..., description="Grade level"
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ..., description="Overall difficulty level"
    )
    content: Optional[str] = Field(
        None, description="Additional content or requirements for the exam"
    )
    total_questions: int = Field(..., ge=1, description="Total number of questions")
    total_points: int = Field(..., ge=1, description="Total points for the exam")
    question_types: Optional[
        List[
            Literal[
                "multiple_choice", "true_false", "fill_blank", "long_answer", "matching"
            ]
        ]
    ] = Field(None, description="Preferred question types")
    provider: str = Field(default="gemini", description="LLM provider")
    model: str = Field(
        default="gemini-2.0-flash-exp", description="LLM model to use"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for prompt rendering."""
        return {
            "topic": self.topic,
            "grade_level": self.grade_level,
            "difficulty": self.difficulty,
            "content": self.content or "",
            "total_questions": self.total_questions,
            "total_points": self.total_points,
            "question_types": (
                ", ".join(self.question_types) if self.question_types else "any"
            ),
        }


class QuestionContext(BaseModel):
    """Context for questions (reading passage, image, etc.)."""

    context_type: Literal["reading_passage", "image", "audio", "video"]
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class MultipleChoiceQuestion(BaseModel):
    """Multiple choice question."""

    question_type: Literal["multiple_choice"] = "multiple_choice"
    content: str = Field(..., description="Question text")
    answers: List[str] = Field(..., min_length=2, description="Answer options")
    correct_answer: str = Field(..., description="Correct answer")
    explanation: Optional[str] = Field(None, description="Explanation of the answer")


class TrueFalseQuestion(BaseModel):
    """True/False question."""

    question_type: Literal["true_false"] = "true_false"
    content: str = Field(..., description="Question text")
    correct_answer: bool = Field(..., description="Correct answer (true or false)")
    explanation: Optional[str] = Field(None, description="Explanation of the answer")


class FillBlankQuestion(BaseModel):
    """Fill in the blank question."""

    question_type: Literal["fill_blank"] = "fill_blank"
    content: str = Field(
        ..., description="Question text with ____ for the blank(s)"
    )
    correct_answer: str = Field(..., description="Correct answer to fill the blank")
    explanation: Optional[str] = Field(None, description="Explanation of the answer")


class LongAnswerQuestion(BaseModel):
    """Long answer/essay question."""

    question_type: Literal["long_answer"] = "long_answer"
    content: str = Field(..., description="Question text")
    correct_answer: str = Field(
        ..., description="Sample correct answer or key points"
    )
    explanation: Optional[str] = Field(None, description="Grading guidelines")


class MatchingPair(BaseModel):
    """A pair for matching questions."""

    left: str
    right: str


class MatchingQuestion(BaseModel):
    """Matching question."""

    question_type: Literal["matching"] = "matching"
    content: str = Field(..., description="Question instruction")
    left: List[str] = Field(..., description="Left side items")
    right: List[str] = Field(..., description="Right side items")
    correct_answer: List[MatchingPair] = Field(
        ..., description="Correct matching pairs"
    )
    explanation: Optional[str] = Field(None, description="Explanation")


Question = Union[
    MultipleChoiceQuestion,
    TrueFalseQuestion,
    FillBlankQuestion,
    LongAnswerQuestion,
    MatchingQuestion,
]


class QuestionWithContext(BaseModel):
    """Question with optional context."""

    context: Optional[QuestionContext] = None
    question_number: Optional[int] = Field(
        None, description="Order within context (1, 2, 3...)"
    )
    topic: str
    grade_level: Literal["K", "1", "2", "3", "4", "5"]
    difficulty: Literal["easy", "medium", "hard"]
    question: Question
    default_points: int = Field(default=1, ge=1)


class GenerateQuestionsRequest(BaseModel):
    """Request to generate questions from a matrix."""

    matrix: List[MatrixItem] = Field(..., description="Exam matrix to generate from")
    provider: str = Field(default="gemini", description="LLM provider")
    model: str = Field(
        default="gemini-2.0-flash-exp", description="LLM model to use"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for prompt rendering."""
        return {
            "matrix_items": [item.model_dump() for item in self.matrix],
            "total_questions": sum(item.count for item in self.matrix),
        }


class GenerationProgress(BaseModel):
    """Progress update for question generation."""

    current: int = Field(..., description="Current progress")
    total: int = Field(..., description="Total items")
    message: str = Field(..., description="Progress message")


class QuestionGenerationStatus(BaseModel):
    """Status response for question generation."""

    status: Literal["GENERATING", "COMPLETED", "ERROR"]
    progress: Optional[GenerationProgress] = None
    questions: Optional[List[QuestionWithContext]] = None
    error: Optional[str] = None
