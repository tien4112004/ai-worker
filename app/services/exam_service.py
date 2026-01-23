"""Service for exam and question generation."""

import json
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.llms.executor import LLMExecutor
from app.prompts.loader import PromptStore
from app.schemas.exam_content import (
    ExamMatrix,
    ExamMatrixV2,
    GenerateMatrixRequest,
    GenerateMatrixV2Request,
    GenerateQuestionsRequest,
    GenerationProgress,
    MatrixCell,
    MatrixContent,
    MatrixDimensions,
    MatrixItem,
    MatrixMetadata,
    DimensionTopic,
    QuestionGenerationStatus,
    QuestionWithContext,
    Topic,
)


class ExamService:
    """Service for generating exams and questions using AI."""

    def __init__(self, llm_executor: LLMExecutor, prompt_store: PromptStore):
        self.llm_executor = llm_executor or LLMExecutor()
        self.prompt_store = prompt_store or PromptStore()

    def _system(self, key: str, vars: Dict[str, Any] | None) -> str:
        """Render a system prompt from the prompt store."""
        return self.prompt_store.render(key, vars)

    def _convert_matrix_items_to_exam_matrix(
        self, items: List[MatrixItem], request: GenerateMatrixRequest
    ) -> ExamMatrix:
        """Convert legacy MatrixItem list to new ExamMatrix structure."""
        # Extract unique topics
        unique_topics = {}
        for item in items:
            if item.topic not in unique_topics:
                unique_topics[item.topic] = Topic(
                    id=str(uuid.uuid4()),
                    name=item.topic,
                    description=None
                )
        
        # Group by difficulty to create contents
        difficulty_groups = {"easy": 0, "medium": 0, "hard": 0}
        for item in items:
            difficulty_groups[item.difficulty] += item.count
        
        contents = [
            MatrixContent(
                difficulty=difficulty,
                number_of_questions=count,
                selected_questions=None
            )
            for difficulty, count in difficulty_groups.items()
            if count > 0
        ]
        
        return ExamMatrix(
            id=str(uuid.uuid4()),
            name=f"{request.topic} - Grade {request.grade_level}",
            description=request.content,
            subject_code=request.topic,  # Using topic as subject code for now
            target_total_points=request.total_points,
            topics=list(unique_topics.values()),
            contents=contents,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            created_by=None
        )

    # Matrix Generation
    def generate_matrix(self, request: GenerateMatrixRequest) -> ExamMatrix:
        """
        Generate an exam matrix using LLM.

        Args:
            request: Request containing exam requirements

        Returns:
            ExamMatrix object representing the exam structure
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
            matrix_items = [MatrixItem(**item) for item in matrix_data]
            return self._convert_matrix_items_to_exam_matrix(matrix_items, request)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse matrix response: {e}\nResponse: {result}")

    # V2 Matrix Generation - 3D Format
    def generate_matrix_v2(self, request: GenerateMatrixV2Request) -> ExamMatrixV2:
        """
        Generate a 3D exam matrix using LLM.

        The matrix is indexed as: matrix[topic_index][difficulty_index][question_type_index]
        Each cell contains {count, points} for that combination.

        Args:
            request: Request containing exam requirements with topics list

        Returns:
            ExamMatrixV2 object representing the 3D exam structure
        """
        # Prepare prompt variables
        prompt_vars = request.to_dict()
        prompt_vars["difficulties"] = request.difficulties or ["easy", "medium", "hard"]
        prompt_vars["question_types"] = request.questionTypes or [
            "multiple_choice", "fill_in_blank", "true_false", "matching"
        ]
        
        sys_msg = self._system("exam.matrix.v2.system", prompt_vars)
        usr_msg = self._system("exam.matrix.v2.user", prompt_vars)

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
            result_text = self._extract_json(result)
            matrix_data = json.loads(result_text)
            
            # Parse into ExamMatrixV2 structure
            metadata = MatrixMetadata(
                id=matrix_data.get("metadata", {}).get("id", str(uuid.uuid4())),
                name=matrix_data.get("metadata", {}).get("name", request.name),
                created_at=matrix_data.get("metadata", {}).get("createdAt", datetime.utcnow().isoformat())
            )
            
            # Parse dimensions
            dims_data = matrix_data.get("dimensions", {})
            topics = [
                DimensionTopic(id=t.get("id", str(uuid.uuid4())), name=t.get("name", "Unknown"))
                for t in dims_data.get("topics", [])
            ]
            
            dimensions = MatrixDimensions(
                topics=topics,
                difficulties=dims_data.get("difficulties", ["easy", "medium", "hard"]),
                question_types=dims_data.get("questionTypes", ["multiple_choice", "fill_in_blank", "true_false", "matching"])
            )
            
            # Parse 3D matrix - expect [count, points] arrays
            raw_matrix = matrix_data.get("matrix", [])
            parsed_matrix = []
            for topic_row in raw_matrix:
                diff_rows = []
                for diff_row in topic_row:
                    qtype_cells = []
                    for cell in diff_row:
                        # Handle both array format [count, points] and object format {count, points}
                        if isinstance(cell, list):
                            qtype_cells.append([int(cell[0]), float(cell[1])])
                        else:
                            qtype_cells.append([cell.get("count", 0), cell.get("points", 0)])
                    diff_rows.append(qtype_cells)
                parsed_matrix.append(diff_rows)
            
            return ExamMatrixV2(
                metadata=metadata,
                dimensions=dimensions,
                matrix=parsed_matrix
            )
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse V2 matrix response: {e}\nResponse: {result}")
        except Exception as e:
            raise ValueError(f"Failed to create V2 matrix: {e}")

    def _extract_json(self, result: str) -> str:
        """Extract JSON from potential markdown code blocks."""
        result_text = result.strip()
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            result_text = "\n".join(
                line
                for line in lines
                if not line.strip().startswith("```")
            )
        return result_text

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
    ) -> ExamMatrix:
        """Generate a mock exam matrix for testing."""
        matrix_items = [
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
        return self._convert_matrix_items_to_exam_matrix(matrix_items, request)

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

    def generate_matrix_v2_mock(
        self, request: GenerateMatrixV2Request
    ) -> ExamMatrixV2:
        """Generate a mock 3D exam matrix for testing without LLM."""
        import uuid as uuid_lib
        
        topics = [
            DimensionTopic(id=str(uuid_lib.uuid4()), name=t)
            for t in request.topics
        ]
        difficulties = request.difficulties or ["easy", "medium", "hard"]
        question_types = request.questionTypes or ["multiple_choice", "fill_in_blank", "true_false", "matching"]
        
        # Build mock 3D matrix with distributed questions
        total_q = request.totalQuestions
        total_p = request.totalPoints
        
        # Distribute questions across cells
        num_topics = len(topics)
        num_diffs = len(difficulties)
        num_qtypes = len(question_types)
        total_cells = num_topics * num_diffs * num_qtypes
        
        base_count = total_q // total_cells if total_cells > 0 else 0
        base_points = total_p / total_q if total_q > 0 else 1
        
        matrix = []
        remaining_q = total_q
        
        for t_idx in range(num_topics):
            topic_rows = []
            for d_idx in range(num_diffs):
                diff_cells = []
                for qt_idx in range(num_qtypes):
                    # Give more questions to easy/medium, fewer to hard
                    modifier = 1.5 if d_idx == 0 else (1.0 if d_idx == 1 else 0.5)
                    count = max(0, min(remaining_q, int(base_count * modifier)))
                    
                    # Last cell gets remaining
                    if t_idx == num_topics - 1 and d_idx == num_diffs - 1 and qt_idx == num_qtypes - 1:
                        count = remaining_q
                    
                    points = count * base_points * (1 + d_idx * 0.5)  # Harder = more points
                    remaining_q -= count
                    
                    # Use [count, points] array format
                    diff_cells.append([count, round(points, 1)])
                topic_rows.append(diff_cells)
            matrix.append(topic_rows)
        
        return ExamMatrixV2(
            metadata=MatrixMetadata(
                id=str(uuid_lib.uuid4()),
                name=request.name,
                created_at=datetime.utcnow().isoformat()
            ),
            dimensions=MatrixDimensions(
                topics=topics,
                difficulties=difficulties,
                question_types=question_types
            ),
            matrix=matrix
        )
