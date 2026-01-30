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


class MultipleChoiceOption(BaseModel):
    """Option for multiple choice question."""
    text: str
    imageUrl: Optional[str] = Field(None, alias="image_url")
    isCorrect: bool = Field(..., alias="is_correct")
    
    class Config:
        populate_by_name = True


class MultipleChoiceData(BaseModel):
    """Data for multiple choice question."""
    type: Literal["MULTIPLE_CHOICE"] = Field(default="MULTIPLE_CHOICE", description="Question type discriminator")
    options: List[MultipleChoiceOption] = Field(..., min_length=4, max_length=4)
    shuffleOptions: bool = Field(True, alias="shuffle_options")
    
    class Config:
        populate_by_name = True


class BlankSegment(BaseModel):
    """Segment for fill in the blank question."""
    type: Literal["TEXT", "BLANK"]
    content: str = Field(default="")
    acceptableAnswers: Optional[List[str]] = Field(None, alias="acceptable_answers")
    
    class Config:
        populate_by_name = True


class FillInBlankData(BaseModel):
    """Data for fill in the blank question."""
    type: Literal["FILL_IN_BLANK"] = Field(default="FILL_IN_BLANK", description="Question type discriminator")
    segments: List[BlankSegment]
    caseSensitive: bool = Field(False, alias="case_sensitive")
    
    class Config:
        populate_by_name = True


class MatchingPair(BaseModel):
    """Pair for matching question."""
    left: str
    leftImageUrl: Optional[str] = Field(None, alias="left_image_url")
    right: str
    rightImageUrl: Optional[str] = Field(None, alias="right_image_url")
    
    class Config:
        populate_by_name = True


class MatchingData(BaseModel):
    """Data for matching question."""
    type: Literal["MATCHING"] = Field(default="MATCHING", description="Question type discriminator")
    pairs: List[MatchingPair] = Field(..., min_length=4)
    shufflePairs: bool = Field(True, alias="shuffle_pairs")
    
    class Config:
        populate_by_name = True


class OpenEndedData(BaseModel):
    """Data for open ended question."""
    type: Literal["OPEN_ENDED"] = Field(default="OPEN_ENDED", description="Question type discriminator")
    expectedAnswer: str = Field(..., alias="expected_answer")
    maxLength: Optional[int] = Field(500, alias="max_length")
    
    class Config:
        populate_by_name = True


class Question(BaseModel):
    """Question entity matching backend Question class."""
    
    type: Literal["MULTIPLE_CHOICE", "FILL_IN_BLANK", "MATCHING", "OPEN_ENDED"]
    difficulty: Literal["KNOWLEDGE", "COMPREHENSION", "APPLICATION", "ADVANCED_APPLICATION"]
    title: str = Field(..., description="Question text/prompt")
    titleImageUrl: Optional[str] = Field(None, alias="title_image_url")
    explanation: Optional[str] = None
    grade: Literal["K", "1", "2", "3", "4", "5"]
    chapter: str = Field(..., description="Topic/chapter name")
    subject: str = Field(..., description="Subject code: T, TV, TA")
    data: Union[MultipleChoiceData, FillInBlankData, MatchingData, OpenEndedData]
    point: float = Field(default=1.0, ge=0)
    
    class Config:
        populate_by_name = True


class GenerateQuestionsRequest(BaseModel):
    """Request to generate questions from a matrix."""

    matrix: List[MatrixItem] = Field(..., description="Exam matrix to generate from")
    provider: str = Field(default="gemini", description="LLM provider")
    model: str = Field(
        default="gemini-2.5-flash", description="LLM model to use"
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
    questions: Optional[List[Question]] = None
    error: Optional[str] = None


class GenerateQuestionsFromTopicRequest(BaseModel):
    """Request to generate questions from a topic."""
    
    topic: str = Field(..., description="Topic or chapter name")
    grade_level: Literal["K", "1", "2", "3", "4", "5"]
    subject_code: str = Field(..., description="Subject code: T, TV, TA")
    
    questions_per_difficulty: Dict[Literal["EASY", "MEDIUM", "HARD"], int] = Field(
        ..., 
        description="Number of questions for each difficulty level"
    )
    
    question_types: List[Literal["MULTIPLE_CHOICE", "FILL_IN_BLANK", "MATCHING", "OPEN_ENDED"]] = Field(
        ...,
        description="Types of questions to generate"
    )
    
    additional_requirements: Optional[str] = Field(
        None,
        description="Additional requirements or context for question generation"
    )
    
    # LLM configuration
    provider: Optional[str] = Field(default="google", description="LLM provider")
    model: Optional[str] = Field(default="gemini-2.5-flash", description="LLM model")
