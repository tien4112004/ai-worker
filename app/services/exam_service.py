"""Service for exam and question generation."""

import json
from typing import Any, AsyncGenerator, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.llms.executor import LLMExecutor
from app.prompts.loader import PromptStore
from app.schemas.exam_content import (
    GenerateMatrixRequest,
    GenerateQuestionsRequest,
    GenerationProgress,
    MatrixItem,
    QuestionGenerationStatus,
    QuestionWithContext,
)


class ExamService:
    """Service for generating exams and questions using AI."""

    def __init__(self, llm_executor: LLMExecutor, prompt_store: PromptStore):
        self.llm_executor = llm_executor or LLMExecutor()
        self.prompt_store = prompt_store or PromptStore()

    def _system(self, key: str, vars: Dict[str, Any] | None) -> str:
        """Render a system prompt from the prompt store."""
        return self.prompt_store.render(key, vars)

    # Matrix Generation
    def generate_matrix(self, request: GenerateMatrixRequest) -> List[MatrixItem]:
        """
        Generate an exam matrix using LLM.

        Args:
            request: Request containing exam requirements

        Returns:
            List of MatrixItem objects representing the exam structure
        """
        sys_msg = self._system("exam.matrix.system", None)
        usr_msg = self._system("exam.matrix.user", request.to_dict())

        result = self.llm_executor.batch(
            provider=request.provider,
            model=request.model,
            messages=[
                SystemMessage(content=sys_msg),
                HumanMessage(content=usr_msg),
            ],
        )

        # Parse the JSON response
        try:
            # Extract JSON from potential markdown code blocks
            result_text = result.strip()
            if result_text.startswith("```"):
                # Remove markdown code block markers
                lines = result_text.split("\n")
                result_text = "\n".join(
                    line
                    for line in lines
                    if not line.strip().startswith("```")
                )

            matrix_data = json.loads(result_text)
            return [MatrixItem(**item) for item in matrix_data]
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse matrix response: {e}\nResponse: {result}")

    async def generate_matrix_stream(
        self, request: GenerateMatrixRequest
    ) -> AsyncGenerator[str, None]:
        """
        Generate an exam matrix using LLM with streaming.

        Args:
            request: Request containing exam requirements

        Yields:
            Chunks of the generated matrix as they arrive
        """
        sys_msg = self._system("exam.matrix.system", None)
        usr_msg = self._system("exam.matrix.user", request.to_dict())

        result = self.llm_executor.stream(
            provider=request.provider,
            model=request.model,
            messages=[
                SystemMessage(content=sys_msg),
                HumanMessage(content=usr_msg),
            ],
        )

        for chunk in result:
            yield chunk

    # Question Generation from Matrix (Priority 1)
    def generate_questions_from_matrix(
        self, request: GenerateQuestionsRequest
    ) -> List[QuestionWithContext]:
        """
        Generate questions based on an exam matrix.

        Args:
            request: Request containing the matrix and generation parameters

        Returns:
            List of generated questions with their contexts
        """
        sys_msg = self._system("exam.questions.system", None)
        usr_msg = self._system("exam.questions.user", request.to_dict())

        result = self.llm_executor.batch(
            provider=request.provider,
            model=request.model,
            messages=[
                SystemMessage(content=sys_msg),
                HumanMessage(content=usr_msg),
            ],
        )

        # Parse the JSON response
        try:
            # Extract JSON from potential markdown code blocks
            result_text = result.strip()
            if result_text.startswith("```"):
                lines = result_text.split("\n")
                result_text = "\n".join(
                    line
                    for line in lines
                    if not line.strip().startswith("```")
                )

            questions_data = json.loads(result_text)

            # Transform the response into QuestionWithContext objects
            all_questions = []
            for item_data in questions_data:
                context = (
                    item_data.get("context") if item_data.get("context") else None
                )

                for q in item_data.get("questions", []):
                    question_obj = QuestionWithContext(
                        context=context,
                        question_number=q.get("question_number"),
                        topic=q.get("topic"),
                        grade_level=q.get("grade_level"),
                        difficulty=q.get("difficulty"),
                        question=q,  # The question dict will be validated by the union type
                        default_points=q.get("default_points", 1),
                    )
                    all_questions.append(question_obj)

            return all_questions
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse questions response: {e}\nResponse: {result}"
            )
        except Exception as e:
            raise ValueError(f"Failed to process questions: {e}")

    async def generate_questions_from_matrix_stream(
        self, request: GenerateQuestionsRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate questions from matrix with streaming progress updates.

        Args:
            request: Request containing the matrix

        Yields:
            Progress updates and generated questions
        """
        sys_msg = self._system("exam.questions.system", None)
        usr_msg = self._system("exam.questions.user", request.to_dict())

        total_questions = sum(item.count for item in request.matrix)

        # Send initial progress
        yield {
            "status": "GENERATING",
            "progress": {
                "current": 0,
                "total": total_questions,
                "message": "Starting question generation...",
            },
        }

        # Stream the LLM response
        accumulated_response = ""
        result_stream = self.llm_executor.stream(
            provider=request.provider,
            model=request.model,
            messages=[
                SystemMessage(content=sys_msg),
                HumanMessage(content=usr_msg),
            ],
        )

        for chunk in result_stream:
            accumulated_response += chunk

            # Try to parse partial JSON to show progress
            # This is a simplified approach - in production you might want more sophisticated parsing
            try:
                # Count how many complete question objects we've received
                question_count = accumulated_response.count('"question_type"')
                if question_count > 0:
                    yield {
                        "status": "GENERATING",
                        "progress": {
                            "current": min(question_count, total_questions),
                            "total": total_questions,
                            "message": f"Generated {question_count} of {total_questions} questions...",
                        },
                    }
            except:
                pass  # Continue if we can't parse partial response

        # Parse final response
        try:
            result_text = accumulated_response.strip()
            if result_text.startswith("```"):
                lines = result_text.split("\n")
                result_text = "\n".join(
                    line
                    for line in lines
                    if not line.strip().startswith("```")
                )

            questions_data = json.loads(result_text)

            # Transform the response
            all_questions = []
            for item_data in questions_data:
                context = (
                    item_data.get("context") if item_data.get("context") else None
                )

                for q in item_data.get("questions", []):
                    question_obj = QuestionWithContext(
                        context=context,
                        question_number=q.get("question_number"),
                        topic=q.get("topic"),
                        grade_level=q.get("grade_level"),
                        difficulty=q.get("difficulty"),
                        question=q,
                        default_points=q.get("default_points", 1),
                    )
                    all_questions.append(question_obj)

            # Send completion with questions
            yield {
                "status": "COMPLETED",
                "progress": {
                    "current": total_questions,
                    "total": total_questions,
                    "message": "Question generation completed!",
                },
                "questions": [q.model_dump() for q in all_questions],
            }

        except Exception as e:
            yield {
                "status": "ERROR",
                "error": f"Failed to process questions: {str(e)}",
            }

    # Mock methods for testing
    def generate_matrix_mock(
        self, request: GenerateMatrixRequest
    ) -> List[MatrixItem]:
        """Generate a mock exam matrix for testing."""
        return [
            MatrixItem(
                topic="Basic Addition",
                question_type="multiple_choice",
                count=3,
                points_each=2,
                difficulty="easy",
                requires_context=False,
            ),
            MatrixItem(
                topic="Number Recognition",
                question_type="true_false",
                count=2,
                points_each=1,
                difficulty="easy",
                requires_context=False,
            ),
        ]

    async def generate_matrix_stream_mock(
        self, request: GenerateMatrixRequest
    ) -> AsyncGenerator[str, None]:
        """Generate a mock matrix stream for testing."""
        mock_matrix = [
            {
                "topic": "Basic Addition",
                "question_type": "multiple_choice",
                "count": 3,
                "points_each": 2,
                "difficulty": "easy",
                "requires_context": False,
            },
            {
                "topic": "Number Recognition",
                "question_type": "true_false",
                "count": 2,
                "points_each": 1,
                "difficulty": "easy",
                "requires_context": False,
            },
        ]

        result = json.dumps(mock_matrix, indent=2)
        # Simulate streaming by yielding chunks
        chunk_size = 50
        for i in range(0, len(result), chunk_size):
            yield result[i : i + chunk_size]

    def generate_questions_mock(
        self, request: GenerateQuestionsRequest
    ) -> List[QuestionWithContext]:
        """Generate mock questions for testing."""
        return [
            QuestionWithContext(
                context=None,
                question_number=None,
                topic="Basic Addition",
                grade_level="1",
                difficulty="easy",
                question={
                    "question_type": "multiple_choice",
                    "content": "What is 2 + 2?",
                    "answers": ["2", "3", "4", "5"],
                    "correct_answer": "4",
                    "explanation": "2 plus 2 equals 4",
                },
                default_points=2,
            )
        ]

    async def generate_questions_stream_mock(
        self, request: GenerateQuestionsRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate mock question stream for testing."""
        total = sum(item.count for item in request.matrix)

        yield {
            "status": "GENERATING",
            "progress": {"current": 0, "total": total, "message": "Starting..."},
        }

        yield {
            "status": "GENERATING",
            "progress": {
                "current": total // 2,
                "total": total,
                "message": f"Generated {total//2} questions...",
            },
        }

        yield {
            "status": "COMPLETED",
            "progress": {
                "current": total,
                "total": total,
                "message": "Completed!",
            },
            "questions": [
                {
                    "topic": "Basic Addition",
                    "grade_level": "1",
                    "difficulty": "easy",
                    "question": {
                        "question_type": "multiple_choice",
                        "content": "What is 2 + 2?",
                        "answers": ["2", "3", "4", "5"],
                        "correct_answer": "4",
                    },
                    "default_points": 2,
                }
            ],
        }
