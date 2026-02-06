import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Generator

from langchain_core.messages import HumanMessage, SystemMessage

from app.llms.executor import LLMExecutor
from app.prompts.loader import PromptStore
from app.prompts.subject_prompt_router import get_subject_grade_prompt_key
from app.schemas.exam_content import (
    DimensionTopic,
    ExamMatrix,
    GenerateMatrixRequest,
    GenerateQuestionsFromTopicRequest,
    MatrixDimensions,
    MatrixMetadata,
    Question,
)
from app.schemas.mindmap_content import MindmapGenerateRequest
from app.schemas.slide_content import (
    OutlineGenerateRequest,
    PresentationGenerateRequest,
)
from app.schemas.token_usage import TokenUsage


class ContentMismatchError(Exception):
    """Raised when retrieved documents do not match the requested topic/subject/grade."""

    pass


class ContentRagService:
    def __init__(self, llm_executor: LLMExecutor, prompt_store: PromptStore):
        self.llm_executor = llm_executor or LLMExecutor()
        self.prompt_store = prompt_store or PromptStore()
        self.last_token_usage = None

    def _system(self, key: str, vars: Dict[str, Any] | None) -> str:
        return self.prompt_store.render(key, vars)

    def _system_with_subject_grade(
        self,
        key: str,
        vars: Dict[str, Any] | None,
        subject_code: str | None,
        grade: str | None,
    ) -> str:
        """Render system prompt with subject-grade specific prompt injected via placeholder.

        Args:
            key: The base prompt key to render
            vars: Variables for template substitution
            subject_code: Optional subject code (e.g., 'T', 'TV', 'TA')
            grade: Optional grade level (e.g., '1', '2', '3', '4', '5')

        Returns:
            System prompt with subject-grade prompt injected if applicable
        """
        # Initialize vars dict if None
        if vars is None:
            vars = {}
        else:
            # Make a copy to avoid mutating the input
            vars = vars.copy()

        # Get subject-grade prompt if both subject and grade are provided
        subject_grade_prompt = ""
        subject_grade_key = get_subject_grade_prompt_key(subject_code, grade)
        if subject_grade_key:
            try:
                subject_grade_prompt = self.prompt_store.render(
                    subject_grade_key, None
                )
            except KeyError:
                print(
                    f"[WARNING] Subject-grade prompt key '{subject_grade_key}' not found in registry"
                )

        # Inject the subject-grade prompt into the template variables
        vars["subject_grade_prompt"] = subject_grade_prompt

        # Render the base template with the injected subject-grade prompt
        return self.prompt_store.render(key, vars)

    @staticmethod
    def _check_content_mismatch(result: dict) -> None:
        answer = result.get("answer", "").strip()
        if answer.startswith("CONTENT_MISMATCH:"):
            message = answer[len("CONTENT_MISMATCH:") :].strip()
            raise ContentMismatchError(message)

    def make_outline_with_rag(self, request: OutlineGenerateRequest):
        """Generate outline using LLM and save result.
        Args:
            request (OutlineGenerateRequest): Request object containing parameters for outline generation.
        Returns:
            Dict: A dictionary containing the generated outline.
        """
        sys_msg = self._system_with_subject_grade(
            "outline.system.rag",
            None,
            request.subject,
            request.grade,
        )

        usr_msg = self._system(
            "outline.user",
            request.to_dict(),
        )

        # Build filters for document search
        # Note: Use subject_code (e.g., 'TV', 'T', 'TA') instead of subject name
        filters = {}
        if request.subject:
            filters["subject_code"] = request.subject
        if request.grade:
            # Convert grade to integer if it's numeric (metadata stores it as int)
            try:
                filters["grade"] = int(request.grade)
            except (ValueError, TypeError):
                filters["grade"] = request.grade

        print(f"[DEBUG] RAG filters being applied: {filters}")
        print(
            f"[DEBUG] Request - subject: {request.subject}, grade: {request.grade} (type: {type(request.grade).__name__})"
        )

        result, token_usage = self.llm_executor.rag_batch(
            provider=request.provider,
            model=request.model,
            query=usr_msg,
            system_prompt=sys_msg,
            return_source_documents=True,
            filters=filters if filters else None,
            custom_prompt=None,
            verbose=False,
        )

        # Store token usage for later access
        self.last_token_usage = token_usage
        self._check_content_mismatch(result)
        return result["answer"]

    def make_presentation_with_rag(self, request: PresentationGenerateRequest):
        sys_msg = self._system_with_subject_grade(
            "presentation.system.rag",
            None,
            request.subject,
            request.grade,
        )

        usr_msg = self._system(
            "presentation.user",
            request.to_dict(),
        )

        filters = {}
        if request.subject:
            filters["subject_code"] = request.subject
        if request.grade:
            try:
                filters["grade"] = int(request.grade)
            except (ValueError, TypeError):
                filters["grade"] = request.grade

        print(f"[DEBUG] RAG filters being applied: {filters}")
        print(
            f"[DEBUG] Request - subject: {request.subject}, grade: {request.grade} (type: {type(request.grade).__name__})"
        )

        result, token_usage = self.llm_executor.rag_batch(
            provider=request.provider,
            model=request.model,
            query=usr_msg,
            system_prompt=sys_msg,
            return_source_documents=True,
            filters=filters if filters else None,
            custom_prompt=None,
            verbose=False,
        )

        self.last_token_usage = token_usage
        self._check_content_mismatch(result)
        return result["answer"]

    def _checked_rag_stream(
        self,
        provider: str,
        model: str,
        query: str,
        system_prompt: str,
        filters,
    ) -> Generator:
        """Start a RAG stream with eager CONTENT_MISMATCH detection.

        Prefetches enough of the stream to check for CONTENT_MISMATCH before
        any chunks are yielded, so the caller can still raise an HTTP 400
        before the SSE response begins.  Returns a generator that replays the
        prefetched chunks then continues with the rest of the stream.
        """
        stream = self.llm_executor.rag_stream(
            provider=provider,
            model=model,
            query=query,
            system_prompt=system_prompt,
            filters=filters,
        )

        # Eagerly consume enough to detect CONTENT_MISMATCH
        buffer = ""
        prefetch: list = []
        for item in stream:
            if isinstance(item, TokenUsage):
                self.last_token_usage = item
                prefetch.append(item)
                break
            buffer += item
            prefetch.append(item)
            if len(buffer) >= 100:
                break

        if buffer.startswith("CONTENT_MISMATCH:"):
            raise ContentMismatchError(
                buffer[len("CONTENT_MISMATCH:") :].strip()
            )

        def _gen():
            yield from prefetch
            for item in stream:
                if isinstance(item, TokenUsage):
                    self.last_token_usage = item
                yield item

        return _gen()

    def make_outline_rag_stream(
        self, request: OutlineGenerateRequest
    ) -> Generator:
        sys_msg = self._system_with_subject_grade(
            "outline.system.rag", None, request.subject, request.grade
        )
        usr_msg = self._system("outline.user", request.to_dict())

        filters = {}
        if request.subject:
            filters["subject_code"] = request.subject
        if request.grade:
            try:
                filters["grade"] = int(request.grade)
            except (ValueError, TypeError):
                filters["grade"] = request.grade

        return self._checked_rag_stream(
            request.provider,
            request.model,
            usr_msg,
            sys_msg,
            filters if filters else None,
        )

    def make_presentation_rag_stream(
        self, request: PresentationGenerateRequest
    ) -> Generator:
        sys_msg = self._system_with_subject_grade(
            "presentation.system.rag", None, request.subject, request.grade
        )
        usr_msg = self._system("presentation.user", request.to_dict())

        filters = {}
        if request.subject:
            filters["subject_code"] = request.subject
        if request.grade:
            try:
                filters["grade"] = int(request.grade)
            except (ValueError, TypeError):
                filters["grade"] = request.grade

        return self._checked_rag_stream(
            request.provider,
            request.model,
            usr_msg,
            sys_msg,
            filters if filters else None,
        )

    def generate_mindmap_with_rag(self, request: MindmapGenerateRequest):
        sys_msg = self._system_with_subject_grade(
            "mindmap.system.rag",
            request.to_dict(),
            request.subject,
            request.grade,
        )

        usr_msg = self._system(
            "mindmap.user",
            request.to_dict(),
        )

        filters = {}
        if request.subject:
            filters["subject_code"] = request.subject
        if request.grade:
            try:
                filters["grade"] = int(request.grade)
            except (ValueError, TypeError):
                filters["grade"] = request.grade

        print(f"[DEBUG] RAG filters being applied: {filters}")
        print(
            f"[DEBUG] Request - subject: {request.subject}, grade: {request.grade} (type: {type(request.grade).__name__})"
        )

        result, token_usage = self.llm_executor.rag_batch(
            provider=request.provider,
            model=request.model,
            query=usr_msg,
            system_prompt=sys_msg,
            return_source_documents=True,
            filters=filters if filters else None,
            custom_prompt=None,
            verbose=False,
        )

        self.last_token_usage = token_usage
        self._check_content_mismatch(result)
        return result["answer"]

    def generate_mindmap_rag_stream(
        self, request: MindmapGenerateRequest
    ) -> Generator:
        sys_msg = self._system_with_subject_grade(
            "mindmap.system.rag",
            request.to_dict(),
            request.subject,
            request.grade,
        )
        usr_msg = self._system("mindmap.user", request.to_dict())

        filters = {}
        if request.subject:
            filters["subject_code"] = request.subject
        if request.grade:
            try:
                filters["grade"] = int(request.grade)
            except (ValueError, TypeError):
                filters["grade"] = request.grade

        return self._checked_rag_stream(
            request.provider,
            request.model,
            usr_msg,
            sys_msg,
            filters if filters else None,
        )

    def _extract_json(self, result: str) -> str:
        """Extract JSON from potential markdown code blocks."""
        result_text = result.strip()
        code_fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        match = re.search(code_fence_pattern, result_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return result_text

    def generate_matrix_with_rag(
        self, request: GenerateMatrixRequest
    ) -> ExamMatrix:
        sys_msg = self._system_with_subject_grade(
            "exam.matrix.system.rag",
            request.to_dict(),
            request.subject,
            request.grade,
        )
        usr_msg = self._system("exam.matrix.user", request.to_dict())

        filters = {}
        if request.subject:
            filters["subject_code"] = request.subject
        if request.grade:
            try:
                filters["grade"] = int(request.grade)
            except (ValueError, TypeError):
                filters["grade"] = request.grade

        result, token_usage = self.llm_executor.rag_batch(
            provider=request.provider,
            model=request.model,
            query=usr_msg,
            system_prompt=sys_msg,
            return_source_documents=True,
            filters=filters if filters else None,
            custom_prompt=None,
            verbose=False,
        )

        self.last_token_usage = token_usage
        self._check_content_mismatch(result)

        try:
            result_text = self._extract_json(result["answer"])
            matrix_data = json.loads(result_text)

            metadata = MatrixMetadata(
                id=matrix_data.get("metadata", {}).get(
                    "id", str(uuid.uuid4())
                ),
                name=matrix_data.get("metadata", {}).get("name", request.name),
                grade=request.grade,
                subject=request.subject,
                created_at=matrix_data.get("metadata", {}).get(
                    "createdAt", datetime.utcnow().isoformat()
                ),
            )

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
                    "difficulties",
                    ["KNOWLEDGE", "COMPREHENSION", "APPLICATION"],
                ),
                question_types=dims_data.get(
                    "questionTypes",
                    [
                        "MULTIPLE_CHOICE",
                        "FILL_IN_BLANK",
                        "TRUE_FALSE",
                        "MATCHING",
                    ],
                ),
            )

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

        except Exception as e:
            raise ValueError(f"Failed to create matrix with RAG: {e}")

    def generate_questions_with_rag(
        self, request: GenerateQuestionsFromTopicRequest
    ) -> list[Question]:
        total_questions = sum(request.questions_per_difficulty.values())
        if total_questions == 0:
            raise ValueError("Total questions must be greater than 0")

        difficulty_distribution = "\n".join(
            [
                f"  - {difficulty}: {count} questions"
                for difficulty, count in request.questions_per_difficulty.items()
                if count > 0
            ]
        )

        subject_map = {
            "T": "Toán (Mathematics)",
            "TV": "Tiếng Việt (Vietnamese)",
            "TA": "Tiếng Anh (English)",
        }
        subject_name = subject_map.get(request.subject, request.subject)

        question_types_str = ", ".join(request.question_types)
        additional_req = ""
        if request.additional_requirements:
            additional_req = f"\n**Additional Requirements**: {request.additional_requirements}"

        prompt_vars = {
            "topic": request.topic,
            "grade": request.grade,
            "subject": subject_name,
            "total_questions": total_questions,
            "difficulty_distribution": difficulty_distribution,
            "question_types": question_types_str,
            "additional_requirements": additional_req,
        }

        sys_msg = self._system_with_subject_grade(
            "question.system.rag",
            None,
            request.subject,
            request.grade,
        )
        usr_msg = self._system("question.user", prompt_vars)

        filters = {}
        if request.subject:
            filters["subject_code"] = request.subject
        if request.grade:
            try:
                filters["grade"] = int(request.grade)
            except (ValueError, TypeError):
                filters["grade"] = request.grade

        result, token_usage = self.llm_executor.rag_batch(
            provider=request.provider,
            model=request.model,
            query=usr_msg,
            system_prompt=sys_msg,
            return_source_documents=True,
            filters=filters if filters else None,
            custom_prompt=None,
            verbose=False,
        )

        self.last_token_usage = token_usage
        self._check_content_mismatch(result)

        try:
            result_text = self._extract_json(result["answer"])
            questions_data = json.loads(result_text)

            if not isinstance(questions_data, list):
                raise ValueError(
                    f"Expected list of questions, got {type(questions_data)}"
                )

            questions = []
            for i, q in enumerate(questions_data):
                try:
                    question = Question(**q)
                    questions.append(question)
                except Exception as e:
                    print(f"[ERROR] Failed to parse question {i}: {e}")
                    raise ValueError(
                        f"Invalid question format at index {i}: {e}"
                    )

            return questions

        except Exception as e:
            raise ValueError(f"Failed to generate questions with RAG: {e}")
