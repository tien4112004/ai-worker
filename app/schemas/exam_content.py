"""Schemas for exam and question generation."""

from typing import Any, Dict, List, Literal, Optional, Union
from datetime import datetime

from pydantic import BaseModel, Field


class Topic(BaseModel):
    """Represents a topic in the exam matrix."""
    
    id: str = Field(..., description="Unique identifier for the topic")
    name: str = Field(..., description="Name of the topic")
    description: Optional[str] = Field(None, description="Optional description of the topic")


class MatrixContent(BaseModel):
    """Represents a content cell in the exam matrix."""
    
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ..., description="Difficulty level"
    )
    numberOfQuestions: int = Field(..., ge=1, description="Number of questions for this difficulty", alias="number_of_questions")
    selectedQuestions: Optional[List[str]] = Field(
        default=None, 
        description="Question IDs currently selected for this cell (Runtime tracking)",
        alias="selected_questions"
    )
    
    class Config:
        populate_by_name = True


# ============================================================================
# New 3D Matrix Format - [topic][difficulty][question_type]
# ============================================================================

class MatrixCell(BaseModel):
    """A cell in the 3D matrix representing question requirements.
    
    When serialized, this becomes "count:points" string format.
    """
    
    count: int = Field(0, ge=0, description="Number of questions required")
    points: float = Field(0, ge=0, description="Total points for this cell")
    
    def to_string(self) -> str:
        """Convert to 'count:points' string format."""
        return f"{self.count}:{self.points}"
    
    @classmethod
    def from_string(cls, s: str) -> "MatrixCell":
        """Create from 'count:points' string format."""
        parts = s.split(":")
        return cls(count=int(parts[0]), points=float(parts[1]))


class DimensionTopic(BaseModel):
    """Topic dimension item in the matrix."""
    
    id: str = Field(..., description="Unique identifier for the topic")
    name: str = Field(..., description="Display name of the topic")


class MatrixDimensions(BaseModel):
    """Dimensions of the 3D exam matrix."""
    
    topics: List[DimensionTopic] = Field(..., description="List of topics (first dimension)")
    difficulties: List[str] = Field(
        default=["easy", "medium", "hard"],
        description="List of difficulty levels (second dimension)"
    )
    questionTypes: List[str] = Field(
        default=["multiple_choice", "fill_in_blank", "true_false", "matching"],
        description="List of question types (third dimension)",
        alias="question_types"
    )
    
    class Config:
        populate_by_name = True


class MatrixMetadata(BaseModel):
    """Metadata for the exam matrix."""
    
    id: str = Field(..., description="Unique identifier for this matrix")
    name: str = Field(..., description="Matrix name")
    createdAt: Optional[str] = Field(None, description="ISO timestamp of creation", alias="created_at")
    
    class Config:
        populate_by_name = True


class ExamMatrix(BaseModel):
    """
    3D exam matrix structure.
    
    Matrix is indexed as: matrix[topic_index][difficulty_index][question_type_index]
    Each cell is "count:points" string format for that combination.
    """
    
    metadata: MatrixMetadata = Field(..., description="Matrix metadata")
    dimensions: MatrixDimensions = Field(..., description="Matrix dimensions")
    matrix: List[List[List[str]]] = Field(
        ..., 
        description="3D matrix: [topic][difficulty][question_type] -> 'count:points'"
    )
    
    class Config:
        populate_by_name = True
    
    def get_total_questions(self) -> int:
        """Calculate total questions across all cells."""
        total = 0
        for topic_row in self.matrix:
            for diff_row in topic_row:
                for cell_str in diff_row:
                    count, _ = cell_str.split(":")
                    total += int(count)
        return total
    
    def get_total_points(self) -> float:
        """Calculate total points across all cells."""
        total = 0.0
        for topic_row in self.matrix:
            for diff_row in topic_row:
                for cell_str in diff_row:
                    _, points = cell_str.split(":")
                    total += float(points)
        return total
    
    def get_cell(self, topic_idx: int, difficulty_idx: int, qtype_idx: int) -> str:
        """Get a specific cell from the matrix as 'count:points' string."""
        return self.matrix[topic_idx][difficulty_idx][qtype_idx]


class GenerateMatrixRequest(BaseModel):
    """Request to generate a 3D exam matrix using AI."""
    
    name: str = Field(..., description="Name for the exam matrix")
    topics: List[str] = Field(..., min_length=1, description="List of topic names to include")
    gradeLevel: str = Field(..., description="Grade level", alias="grade_level")
    totalQuestions: int = Field(..., ge=1, description="Target total number of questions", alias="total_questions")
    totalPoints: int = Field(..., ge=1, description="Target total points", alias="total_points")
    difficulties: Optional[List[str]] = Field(
        default=["easy", "medium", "hard"],
        description="Difficulty levels to include"
    )
    questionTypes: Optional[List[str]] = Field(
        default=["multiple_choice", "fill_in_blank", "true_false", "matching"],
        description="Question types to include",
        alias="question_types"
    )
    additionalRequirements: Optional[str] = Field(
        None, 
        description="Additional requirements or context for the exam",
        alias="additional_requirements"
    )
    provider: str = Field(default="gemini", description="LLM provider")
    model: str = Field(default="gemini-2.5-flash", description="LLM model to use")
    
    class Config:
        populate_by_name = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for prompt rendering."""
        return {
            "name": self.name,
            "topics": ", ".join(self.topics),
            "topics_list": self.topics,
            "grade_level": self.gradeLevel,
            "total_questions": self.totalQuestions,
            "total_points": self.totalPoints,
            "difficulties": ", ".join(self.difficulties) if self.difficulties else "easy, medium, hard",
            "question_types": ", ".join(self.questionTypes) if self.questionTypes else "multiple_choice, fill_in_blank, true_false, matching",
            "additional_requirements": self.additionalRequirements or "",
        }


class MatrixItem(BaseModel):
    """Represents a single item in the exam matrix (Legacy format)."""

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
