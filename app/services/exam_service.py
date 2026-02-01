"""Service for exam and question generation."""

import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.llms.executor import LLMExecutor
from app.prompts.loader import PromptStore
from app.schemas.exam_content import (
    DimensionTopic,
    ExamMatrix,
    GenerateMatrixRequest,
    GenerateQuestionsFromContextRequest,
    GenerateQuestionsFromTopicRequest,
    GenerateQuestionsRequest,
    GenerationProgress,
    MatrixCell,
    MatrixContent,
    MatrixDimensions,
    MatrixItem,
    MatrixMetadata,
    Question,
    QuestionGenerationStatus,
    Topic,
)

logger = logging.getLogger(__name__)


class ExamService:
    """Service for generating exams and questions using AI."""

    def __init__(self, llm_executor: LLMExecutor, prompt_store: PromptStore):
        self.llm_executor = llm_executor or LLMExecutor()
        self.prompt_store = prompt_store or PromptStore()

    def _system(self, key: str, vars: Dict[str, Any] | None) -> str:
        """Render a system prompt from the prompt store."""
        return self.prompt_store.render(key, vars)

    # Matrix Generation
    def generate_matrix(self, request: GenerateMatrixRequest) -> ExamMatrix:
        """
        Generate an exam matrix using LLM.

        The matrix is indexed as: matrix[topic_index][difficulty_index][question_type_index]
        Each cell contains {count, points} for that combination.

        Args:
            request: Request containing exam requirements with topics list

        Returns:
            ExamMatrix object representing the exam structure
        """
        # Prepare prompt variables
        prompt_vars = request.to_dict()
        prompt_vars["difficulties"] = request.difficulties or [
            "knowledge",
            "comprehension",
            "application",
        ]
        prompt_vars["question_types"] = request.questionTypes or [
            "multiple_choice",
            "fill_in_blank",
            "true_false",
            "matching",
        ]

        sys_msg = self._system("exam.matrix.system", prompt_vars)
        usr_msg = self._system("exam.matrix.user", prompt_vars)

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

            metadata = MatrixMetadata(
                id=matrix_data.get("metadata", {}).get(
                    "id", str(uuid.uuid4())
                ),
                name=matrix_data.get("metadata", {}).get("name", request.name),
                grade=request.gradeLevel,
                subject_code=request.subjectCode,
                created_at=matrix_data.get("metadata", {}).get(
                    "createdAt", datetime.utcnow().isoformat()
                ),
            )

            # Parse dimensions
            dims_data = matrix_data.get("dimensions", {})
            topics = [
                DimensionTopic(
                    id=t.get("id", str(uuid.uuid4())),
                    name=t.get("name", "Unknown"),
                )
                for t in dims_data.get("topics", [])
            ]

            dimensions = MatrixDimensions(
                topics=topics,
                difficulties=dims_data.get(
                    "difficulties", ["easy", "medium", "hard"]
                ),
                question_types=dims_data.get(
                    "questionTypes",
                    [
                        "multiple_choice",
                        "fill_in_blank",
                        "true_false",
                        "matching",
                    ],
                ),
            )

            # Parse matrix - convert to "count:points" string format
            raw_matrix = matrix_data.get("matrix", [])
            parsed_matrix = []
            for topic_row in raw_matrix:
                diff_rows = []
                for diff_row in topic_row:
                    qtype_cells = []
                    for cell in diff_row:
                        if isinstance(cell, str):
                            qtype_cells.append(cell)
                        elif isinstance(cell, list):
                            qtype_cells.append(
                                f"{int(cell[0])}:{float(cell[1])}"
                            )
                        else:
                            qtype_cells.append(
                                f"{cell.get('count', 0)}:{cell.get('points', 0)}"
                            )
                    diff_rows.append(qtype_cells)
                parsed_matrix.append(diff_rows)

            return ExamMatrix(
                metadata=metadata, dimensions=dimensions, matrix=parsed_matrix
            )

        except json.JSONDecodeError as e:
            # Truncate response in error message to prevent overwhelming logs
            response_preview = (
                result[:500] + "..." if len(result) > 500 else result
            )
            raise ValueError(
                f"Failed to parse matrix response: {e}\nResponse preview: {response_preview}"
            )
        except Exception as e:
            raise ValueError(f"Failed to create matrix: {e}")

    def _extract_json(self, result: str) -> str:
        """Extract JSON from potential markdown code blocks.

        Uses regex to robustly extract content between code fences.
        """
        result_text = result.strip()

        # Try to extract JSON from code fences using regex
        # Matches ```json or ``` followed by content and closing ```
        code_fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        match = re.search(code_fence_pattern, result_text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: if no code fences found, return as-is
        return result_text

    # Question Generation from Matrix (Priority 1)
    def generate_questions_from_matrix(
        self, request: GenerateQuestionsRequest
    ) -> List[Question]:
        """
        DEPRECATED: Legacy matrix-based generation.
        Use generate_questions_from_topic instead.
        """
        raise NotImplementedError(
            "Matrix-based question generation is deprecated. "
            "Use /questions/generate endpoint with GenerateQuestionsFromTopicRequest instead."
        )

    # # TODO: Do it later
    #     async def generate_questions_from_matrix_stream(
    #         self, request: GenerateQuestionsRequest
    #     ) -> AsyncGenerator[Dict[str, Any], None]:
    #         """
    #         Generate questions from matrix with streaming progress updates.

    #         Args:
    #             request: Request containing the matrix

    #         Yields:
    #             Progress updates and generated questions
    #         """
    #         sys_msg = self._system("exam.questions.system", None)
    #         usr_msg = self._system("exam.questions.user", request.to_dict())

    #         total_questions = sum(item.count for item in request.matrix)

    #         # Send initial progress
    #         yield {
    #             "status": "GENERATING",
    #             "progress": {
    #                 "current": 0,
    #                 "total": total_questions,
    #                 "message": "Starting question generation...",
    #             },
    #         }

    #         # Stream the LLM response
    #         accumulated_response = ""
    #         result_stream = self.llm_executor.stream(
    #             provider=request.provider,
    #             model=request.model,
    #             messages=[
    #                 SystemMessage(content=sys_msg),
    #                 HumanMessage(content=usr_msg),
    #             ],
    #         )

    #         for chunk in result_stream:
    #             accumulated_response += chunk

    #             # Try to parse partial JSON to show progress
    #             # This is a simplified approach - in production you might want more sophisticated parsing
    #             try:
    #                 # Count how many complete question objects we've received
    #                 question_count = accumulated_response.count('"question_type"')
    #                 if question_count > 0:
    #                     yield {
    #                         "status": "GENERATING",
    #                         "progress": {
    #                             "current": min(question_count, total_questions),
    #                             "total": total_questions,
    #                             "message": f"Generated {question_count} of {total_questions} questions...",
    #                         },
    #                     }
    #             except Exception as e:
    #                 logger.debug(
    #                     f"Could not parse partial streaming response: {e}"
    #                 )
    #                 # Continue streaming - partial parse failures are expected

    #         # Parse final response
    #         try:
    #             result_text = accumulated_response.strip()
    #             if result_text.startswith("```"):
    #                 lines = result_text.split("\n")
    #                 result_text = "\n".join(
    #                     line
    #                     for line in lines
    #                     if not line.strip().startswith("```")
    #                 )

    #             questions_data = json.loads(result_text)

    #             # Transform the response
    #             all_questions = []
    #             for item_data in questions_data:
    #                 context = (
    #                     item_data.get("context")
    #                     if item_data.get("context")
    #                     else None
    #                 )

    #                 for q in item_data.get("questions", []):
    #                     question_obj = QuestionWithContext(
    #                         context=context,
    #                         question_number=q.get("question_number"),
    #                         topic=q.get("topic"),
    #                         grade_level=q.get("grade_level"),
    #                         difficulty=q.get("difficulty"),
    #                         question=q,
    #                         default_points=q.get("default_points", 1),
    #                     )
    #                     all_questions.append(question_obj)

    #             # Send completion with questions
    #             yield {
    #                 "status": "COMPLETED",
    #                 "progress": {
    #                     "current": total_questions,
    #                     "total": total_questions,
    #                     "message": "Question generation completed!",
    #                 },
    #                 "questions": [q.model_dump() for q in all_questions],
    #             }

    #         except Exception as e:
    #             yield {
    #                 "status": "ERROR",
    #                 "error": f"Failed to process questions: {str(e)}",
    #             }

    def generate_questions_mock(
        self, request: GenerateQuestionsRequest
    ) -> List[Question]:
        """DEPRECATED: Generate mock questions for testing."""
        raise NotImplementedError(
            "Matrix-based question generation is deprecated. "
            "Use /questions/generate endpoint instead."
        )

    async def generate_questions_stream_mock(
        self, request: GenerateQuestionsRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate mock question stream for testing."""
        total = sum(item.count for item in request.matrix)

        yield {
            "status": "GENERATING",
            "progress": {
                "current": 0,
                "total": total,
                "message": "Starting...",
            },
        }

        yield {
            "status": "GENERATING",
            "progress": {
                "current": total // 2,
                "total": total,
                "message": f"Generated {total // 2} questions...",
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

    def generate_matrix_mock(
        self, request: GenerateMatrixRequest
    ) -> ExamMatrix:
        """Generate a mock exam matrix for testing without LLM."""
        topics = [
            DimensionTopic(id=str(uuid.uuid4()), name=t)
            for t in request.topics
        ]
        difficulties = request.difficulties or [
            "knowledge",
            "comprehension",
            "application",
        ]
        question_types = request.questionTypes or [
            "multiple_choice",
            "fill_in_blank",
            "true_false",
            "matching",
        ]

        # Build mock matrix with distributed questions
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
                    modifier = (
                        1.5 if d_idx == 0 else (1.0 if d_idx == 1 else 0.5)
                    )
                    count = max(
                        0, min(remaining_q, int(base_count * modifier))
                    )

                    # Last cell gets remaining
                    if (
                        t_idx == num_topics - 1
                        and d_idx == num_diffs - 1
                        and qt_idx == num_qtypes - 1
                    ):
                        count = remaining_q

                    points = (
                        count * base_points * (1 + d_idx * 0.5)
                    )  # Harder = more points
                    remaining_q -= count

                    # Use "count:points" string format
                    diff_cells.append(f"{count}:{round(points, 1)}")
                topic_rows.append(diff_cells)
            matrix.append(topic_rows)

        return ExamMatrix(
            metadata=MatrixMetadata(
                id=str(uuid.uuid4()),
                name=request.name,
                created_at=datetime.utcnow().isoformat(),
            ),
            dimensions=MatrixDimensions(
                topics=topics,
                difficulties=difficulties,
                question_types=question_types,
            ),
            matrix=matrix,
        )

    def generate_questions_from_topic(
        self, request: GenerateQuestionsFromTopicRequest
    ) -> List[Question]:
        """
        Generate questions based on topic and requirements.

        Args:
            request: Request containing topic, grade, subject, and requirements

        Returns:
            List of generated Question objects
        """
        logger.info(
            f"[EXAM_SERVICE] Generating questions for topic: {request.topic}, grade: {request.grade_level}"
        )

        # Calculate total questions
        total_questions = sum(request.questions_per_difficulty.values())

        if total_questions == 0:
            raise ValueError("Total questions must be greater than 0")

        # Format difficulty distribution
        difficulty_distribution = "\n".join(
            [
                f"  - {difficulty.lower()}: {count} questions"
                for difficulty, count in request.questions_per_difficulty.items()
                if count > 0
            ]
        )

        # Map subject codes to names
        subject_map = {
            "T": "Toán (Mathematics)",
            "TV": "Tiếng Việt (Vietnamese)",
            "TA": "Tiếng Anh (English)",
        }
        subject_name = subject_map.get(request.subject_code)
        if not subject_name:
            raise ValueError(f"Unknown subject code: {request.subject_code}")

        # Format question types
        question_types_str = ", ".join(request.question_types)

        # Format additional requirements
        additional_req = ""
        if request.additional_requirements:
            additional_req = f"\n**Additional Requirements**: {request.additional_requirements}"

        # Build prompt variables
        prompt_vars = {
            "topic": request.topic,
            "grade_level": request.grade_level,
            "subject": subject_name,
            "total_questions": total_questions,
            "difficulty_distribution": difficulty_distribution,
            "question_types": question_types_str,
            "additional_requirements": additional_req,
        }

        # Render prompts
        sys_msg = self._system("question.system", prompt_vars)
        usr_msg = self._system("question.user", prompt_vars)

        # Execute LLM call
        logger.info(
            f"[EXAM_SERVICE] Calling LLM with provider: {request.provider}, model: {request.model}"
        )

        result, token_usage = self.llm_executor.batch(
            provider=request.provider or "google",
            model=request.model or "gemini-2.5-flash",
            messages=[
                SystemMessage(content=sys_msg),
                HumanMessage(content=usr_msg),
            ],
        )

        logger.info(
            f"[EXAM_SERVICE] LLM call completed. Tokens: input={token_usage.input_tokens}, output={token_usage.output_tokens}"
        )

        # Parse result
        try:
            result_text = self._extract_json(result)
            questions_data = json.loads(result_text)

            # Validate is list
            if not isinstance(questions_data, list):
                raise ValueError(
                    f"Expected list of questions, got {type(questions_data)}"
                )

            # Validate count
            if len(questions_data) != total_questions:
                logger.warning(
                    f"[EXAM_SERVICE] Expected {total_questions} questions, got {len(questions_data)}"
                )

            # Convert to Question objects with validation
            questions = []
            for i, q in enumerate(questions_data):
                try:
                    question = Question(**q)
                    questions.append(question)
                except Exception as e:
                    logger.error(
                        f"[EXAM_SERVICE] Failed to parse question {i}: {e}"
                    )
                    logger.error(f"[EXAM_SERVICE] Question data: {q}")
                    raise ValueError(
                        f"Invalid question format at index {i}: {e}"
                    )

            logger.info(
                f"[EXAM_SERVICE] Successfully generated {len(questions)} questions"
            )
            return questions

        except json.JSONDecodeError as e:
            logger.error(f"[EXAM_SERVICE] JSON parsing error: {e}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")

    def generate_questions_from_context(
        self, request: GenerateQuestionsFromContextRequest
    ) -> List[Question]:
        """
        Generate questions based on context (image/text) and requirements.

        Args:
            request: Request containing context, topic, grade, subject, and requirements

        Returns:
            List of generated Question objects
        """
        logger.info(
            f"[EXAM_SERVICE] Generating questions from context for grade: {request.grade_level}"
        )

        # Calculate total questions
        total_questions = sum(request.questions_per_difficulty.values())

        if total_questions == 0:
            raise ValueError("Total questions must be greater than 0")

        # Format difficulty distribution
        difficulty_distribution = "\n".join(
            [
                f"  - {difficulty.lower()}: {count} questions"
                for difficulty, count in request.questions_per_difficulty.items()
                if count > 0
            ]
        )

        # Map subject codes to names
        subject_map = {
            "T": "Toán (Mathematics)",
            "TV": "Tiếng Việt (Vietnamese)",
            "TA": "Tiếng Anh (English)",
        }
        subject_name = subject_map.get(request.subject_code)
        if not subject_name:
            raise ValueError(f"Unknown subject code: {request.subject_code}")

        # Format question types
        question_types_str = ", ".join(request.question_types)

        # Format objectives
        objectives_str = "\n".join([f"- {obj}" for obj in request.objectives])

        # Format additional requirements
        additional_req = ""
        if request.additional_requirements:
            additional_req = f"\n**Additional Requirements**: {request.additional_requirements}"

        # Build prompt variables
        prompt_vars = {
            "context_type": request.context_type,
            "objectives": objectives_str,
            "grade_level": request.grade_level,
            "subject": subject_name,
            "total_questions": total_questions,
            "difficulty_distribution": difficulty_distribution,
            "question_types": question_types_str,
            "additional_requirements": additional_req,
        }

        # Render prompt text
        sys_msg_content = self._system("question.context.system", prompt_vars)
        usr_msg_content = self._system("question.context.user", prompt_vars)

        messages = [SystemMessage(content=sys_msg_content)]

        if request.context_type == "IMAGE":
            # For image, we need to construct a multipart message
            # Clean base64 string if needed
            image_data = request.context
            if "," in image_data:
                image_data = image_data.split(",")[1]

            messages.append(
                HumanMessage(
                    content=[
                        {"type": "text", "text": usr_msg_content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ]
                )
            )
        else:
            # For text, append context to user message
            full_usr_msg = (
                f"{usr_msg_content}\n\n**Context**:\n{request.context}"
            )
            messages.append(HumanMessage(content=full_usr_msg))

        # Execute LLM call
        logger.info(
            f"[EXAM_SERVICE] Calling LLM with provider: {request.provider}, model: {request.model}"
        )

        result, token_usage = self.llm_executor.batch(
            provider=request.provider or "google",
            model=request.model or "gemini-2.5-flash",
            messages=messages,
        )

        logger.info(
            f"[EXAM_SERVICE] LLM call completed. Tokens: input={token_usage.input_tokens}, output={token_usage.output_tokens}"
        )

        # Parse result
        try:
            result_text = self._extract_json(result)
            questions_data = json.loads(result_text)

            # Validate is list
            if not isinstance(questions_data, list):
                raise ValueError(
                    f"Expected list of questions, got {type(questions_data)}"
                )

            # Validate count
            if len(questions_data) != total_questions:
                logger.warning(
                    f"[EXAM_SERVICE] Expected {total_questions} questions, got {len(questions_data)}"
                )

            # Convert to Question objects with validation
            questions = []
            for i, q in enumerate(questions_data):
                try:
                    question = Question(**q)
                    questions.append(question)
                except Exception as e:
                    logger.error(
                        f"[EXAM_SERVICE] Failed to parse question {i}: {e}"
                    )
                    logger.error(f"[EXAM_SERVICE] Question data: {q}")
                    raise ValueError(
                        f"Invalid question format at index {i}: {e}"
                    )

            logger.info(
                f"[EXAM_SERVICE] Successfully generated {len(questions)} questions"
            )
            return questions

        except json.JSONDecodeError as e:
            logger.error(f"[EXAM_SERVICE] JSON parsing error: {e}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
