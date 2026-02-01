import base64
import os
import random
from asyncio import sleep
from typing import Any, Dict, Generator, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from app.llms.executor import LLMExecutor
from app.prompts.loader import PromptStore
from app.schemas.image_content import ImageGenerateRequest
from app.schemas.mindmap_content import MindmapGenerateRequest
from app.schemas.slide_content import (
    OutlineGenerateRequest,
    PresentationGenerateRequest,
)
from app.schemas.token_usage import TokenUsage


class ContentService:
    def __init__(self, llm_executor: LLMExecutor, prompt_store: PromptStore):
        self.llm_executor = llm_executor or LLMExecutor()
        self.prompt_store = prompt_store or PromptStore()
        self.last_token_usage = None

    def _system(self, key: str, vars: Dict[str, Any] | None) -> str:
        return self.prompt_store.render(key, vars)

    # Presentation Generation
    def make_presentation_stream(self, request: PresentationGenerateRequest):
        """Generate slide content using LLM and save result.
        Args:
            request (PresentationGenerateRequest): Request object containing parameters for slide generation.
        Returns:
            Tuple: (chunks, token_usage) - list of content chunks and token usage data.
        """
        sys_msg = self._system(
            "presentation.system",
            None,
        )

        usr_msg = self._system(
            "presentation.user",
            request.to_dict(),
        )

        chunks, token_usage = self.llm_executor.stream(
            provider=request.provider,
            model=request.model,
            messages=[
                SystemMessage(content=sys_msg),
                HumanMessage(content=usr_msg),
            ],
        )

        # Filter out token_usage objects (only check last chunk for efficiency)
        import json

        filtered_chunks = chunks[:-1] if chunks else []

        # Only parse the last chunk if it exists
        if chunks:
            last_chunk = chunks[-1]
            if not (
                last_chunk.startswith('{"token_usage"')
                or last_chunk.startswith('{"type":"token_usage"')
            ):
                filtered_chunks.append(last_chunk)

        # Store token usage for later access
        self.last_token_usage = token_usage
        return filtered_chunks, token_usage

    def make_presentation(self, request: PresentationGenerateRequest):
        """
        Generate slide content using LLM and save result.
        Args:
            request (PresentationGenerateRequest): Request object containing parameters for slide generation.
        Returns:
            Dict: A dictionary containing the generated slide content.
        """
        sys_msg = self._system(
            "presentation.system",
            None,
        )

        usr_msg = self._system(
            "presentation.user",
            request.to_dict(),
        )

        result, token_usage = self.llm_executor.batch(
            provider=request.provider,
            model=request.model,
            messages=[
                SystemMessage(content=sys_msg),
                HumanMessage(content=usr_msg),
            ],
        )

        # Store token usage for later access
        self.last_token_usage = token_usage
        return result

    # Outline Generation
    def make_outline_stream(self, request: OutlineGenerateRequest):
        """Generate outline using LLM and save result.
        Args:
            request (OutlineGenerateRequest): Request object containing parameters for outline generation.
        Returns:
            Tuple: (chunks, token_usage) - list of content chunks and token usage data.
        """
        sys_msg = self._system(
            "outline.system",
            None,
        )

        usr_msg = self._system(
            "outline.user",
            request.to_dict(),
        )

        chunks, token_usage = self.llm_executor.stream(
            provider=request.provider,
            model=request.model,
            messages=[
                SystemMessage(content=sys_msg),
                HumanMessage(content=usr_msg),
            ],
        )

        # Store token usage for later access
        self.last_token_usage = token_usage
        return chunks, token_usage

    def make_outline(self, request: OutlineGenerateRequest):
        """Generate outline using LLM and save result.
        Args:
            request (OutlineGenerateRequest): Request object containing parameters for outline generation.
        Returns:
            Dict: A dictionary containing the generated outline.
        """
        sys_msg = self._system(
            "outline.system",
            None,
        )

        usr_msg = self._system(
            "outline.user",
            request.to_dict(),
        )

        result, token_usage = self.llm_executor.batch(
            provider=request.provider,
            model=request.model,
            messages=[
                SystemMessage(content=sys_msg),
                HumanMessage(content=usr_msg),
            ],
        )

        # Store token usage for later access
        self.last_token_usage = token_usage
        return result

    def make_presentation_mock(
        self, request: PresentationGenerateRequest
    ) -> Tuple[str, TokenUsage]:
        """Generate mock slide content for testing purposes.
        Returns:
            Tuple: (result, token_usage) - mock slide content and zero token usage.
        """

        sys_msg = self._system(
            "presentation.system",
            request.to_dict(),
        )
        print("System Prompt:", sys_msg)  # Debug print

        result = '```json\n{\n  "slides": [\n    {\n      "type": "main_image",\n      "data": {\n        "image": "Children looking excitedly at an old map of Vietnam with a river highlighted",\n        "content": "Giới thiệu: Một Cuộc Phiêu Lưu Lịch Sử Về Sông Bạch Đằng!"\n      }\n    },\n    {\n      "type": "two_column_with_image",\n      "title": "Ai Đã Xâm Lược Nước Ta?",\n      "data": {\n        "items": [\n          "Quân địch đến từ nước Nam Hán.",\n          "Họ muốn chiếm đất nước ta.",\n          "Nhân dân ta không muốn bị mất nước."\n        ],\n        "image": "Illustration of ancient Chinese warships sailing towards Vietnamese shores"\n      }\n    },\n    {\n      "type": "two_column_with_image",\n      "title": "\\"Bẫy\\" Trên Sông: Ý Tưởng Của Ngô Quyền!",\n      "data": {\n        "items": [\n          "Ngô Quyền cho cắm cọc nhọn dưới sông.",\n          "Cọc ẩn dưới nước lúc triều lên.",\n          "Nhô lên đâm thủng thuyền địch khi nước rút."\n        ],\n        "image": "Illustration of a wooden stake hidden underwater in a river with a boat approaching"\n      }\n    },\n    {\n      "type": "two_column_with_image",\n      "title": "Trận Chiến Rực Lửa Trên Sông!",\n      "data": {\n        "items": [\n          "Thuyền địch mắc bẫy, bị đâm thủng.",\n          "Quân ta tấn công từ hai bên bờ.",\n          "Chiến thắng vang dội cho dân tộc!"\n        ],\n        "image": "Illustration of Vietnamese soldiers attacking enemy ships from the riverbanks during a battle"\n      }\n    },\n    {\n      "type": "main_image",\n      "data": {\n        "image": "Illustration of a proud Vietnamese flag waving over a peaceful landscape",\n        "content": "Chiến thắng Bạch Đằng giúp đất nước ta mãi mãi tự do!"\n      }\n    }\n  ]\n}\n```'

        # Create zero token usage for mock
        mock_usage = TokenUsage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            model="mock",
            provider="mock",
        )
        self.last_token_usage = mock_usage

        return result, mock_usage

    async def make_presentation_stream_mock(
        self,
    ) -> Tuple[List[Dict], TokenUsage]:
        """Generate mock presentation stream for testing purposes.
        Returns:
            Tuple: (slides, token_usage) - list of slide objects and zero token usage.
        """
        slides = [
            {
                "type": "main_image",
                "data": {
                    "image": "Children looking excitedly at an old map of Vietnam with a river highlighted",
                    "content": "Giới thiệu: Một Cuộc Phiêu Lưu Lịch Sử Về Sông Bạch Đằng!",
                },
            },
            {
                "type": "two_column_with_image",
                "title": "Ai Đã Xâm Lược Nước Ta?",
                "data": {
                    "items": [
                        "Quân địch đến từ nước Nam Hán.",
                        "Họ muốn chiếm đất nước ta.",
                        "Nhân dân ta không muốn bị mất nước.",
                    ],
                    "image": "Illustration of ancient Chinese warships sailing towards Vietnamese shores",
                },
            },
            {
                "type": "two_column_with_image",
                "title": '"Bẫy" Trên Sông: Ý Tưởng Của Ngô Quyền!',
                "data": {
                    "items": [
                        "Ngô Quyền cho cắm cọc nhọn dưới sông.",
                        "Cọc ẩn dưới nước lúc triều lên.",
                        "Nhô lên đâm thủng thuyền địch khi nước rút.",
                    ],
                    "image": "Illustration of a wooden stake hidden underwater in a river with a boat approaching",
                },
            },
            {
                "type": "two_column_with_image",
                "title": "Trận Chiến Rực Lửa Trên Sông!",
                "data": {
                    "items": [
                        "Thuyền địch mắc bẫy, bị đâm thủng.",
                        "Quân ta tấn công từ hai bên bờ.",
                        "Chiến thắng vang dội cho dân tộc!",
                    ],
                    "image": "Illustration of Vietnamese soldiers attacking enemy ships from the riverbanks during a battle",
                },
            },
            {
                "type": "main_image",
                "data": {
                    "image": "Illustration of a proud Vietnamese flag waving over a peaceful landscape",
                    "content": "Chiến thắng Bạch Đằng giúp đất nước ta mãi mãi tự do!",
                },
            },
        ]

        mock_usage = TokenUsage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            model="mock",
            provider="mock",
        )
        self.last_token_usage = mock_usage

        return slides, mock_usage

    def make_outline_stream_mock(self) -> Tuple[List[str], TokenUsage]:
        """Generate mock outline stream for testing purposes.
        Returns:
            Tuple: (chunks, token_usage) - list of text chunks and zero token usage.
        """
        import re

        outline = '### Giới thiệu: Một Cuộc Phiêu Lưu Lịch Sử Về Sông Bạch Đằng!\n\nChào các bạn nhỏ! Hôm nay, chúng ta sẽ cùng nhau du hành về quá khứ, đến với một khúc sông thật đặc biệt, nơi đã diễn ra một trận chiến lừng lẫy, giúp bảo vệ đất nước Việt Nam của chúng ta. Các bạn đã sẵn sàng chưa nào?\n\n*   Chúng ta sẽ khám phá câu chuyện về **Sông Bạch Đằng** – một dòng sông hùng vĩ.\n*   Tìm hiểu về những người anh hùng dũng cảm đã chiến đấu trên dòng sông này.\n*   Và hiểu tại sao trận chiến này lại quan trọng đến vậy!\n\n_Hãy chuẩn bị tinh thần để trở thành những nhà thám hiểm lịch sử nhé!_\n\n### Ai Đã Xâm Lược Nước Ta?\n\nNgày xưa, có những đội quân từ phương Bắc muốn xâm chiếm đất nước ta. Họ rất đông và mạnh mẽ, giống như một cơn bão sắp ập đến vậy.\n\n*   Quân địch đến từ **nước Nam Hán** (nay thuộc Trung Quốc).\n*   Họ muốn chiếm đóng và cai trị đất nước của chúng ta.\n*   Nhân dân ta rất lo sợ, nhưng không hề muốn bị mất nước.\n\n> Tưởng tượng xem, nếu có ai đó muốn lấy đi đồ chơi yêu thích của bạn, bạn sẽ làm gì? Ông cha ta cũng đã rất quyết tâm bảo vệ đất nước mình!\n\n### "Bẫy" Trên Sông: Ý Tưởng Tuyệt Vời Của Ngô Quyền!\n\nĐể chống lại quân địch mạnh mẽ, Ngô Quyền – vị tướng tài ba của chúng ta – đã nghĩ ra một kế hoạch vô cùng thông minh và độc đáo. Đó là sử dụng chính dòng sông Bạch Đằng để làm "chiến trường"!\n\n*   Ngô Quyền cho quân lính **cắm cọc nhọn** xuống lòng sông, ẩn dưới mặt nước lúc triều lên.\n*   Khi **triều rút**, những chiếc cọc này sẽ nhô lên, sẵn sàng đâm thủng thuyền địch.\n*   Đây là một cái bẫy thiên nhiên tuyệt vời!\n\n_Giống như chúng ta giăng bẫy chuột vậy đó, nhưng là bẫy cho thuyền lớn!_\n\n### Trận Chiến Rực Lửa Trên Sông!\n\nKhi quân Nam Hán hùng hổ tiến vào sông Bạch Đằng, họ đã mắc bẫy của Ngô Quyền.\n\n*   Thuyền địch bị **đâm thủng** bởi những chiếc cọc nhọn khi nước rút.\n*   Quân ta từ hai bên bờ sông đã **tấn công dữ dội**.\n*   Trận chiến diễn ra vô cùng ác liệt, nhưng quân ta đã chiến thắng vang dội!\n\n> Tiếng reo hò vang vọng khắp sông, đánh dấu một chiến thắng vẻ vang cho dân tộc!\n\n### Ý Nghĩa Lịch Sử: Vì Sao Chúng Ta Nhớ Mãi?\n\nChiến thắng sông Bạch Đằng không chỉ là một trận đánh hay, mà nó còn mang một ý nghĩa vô cùng to lớn đối với lịch sử Việt Nam.\n\n*   Trận chiến này đã giúp **giải phóng đất nước** khỏi ách đô hộ của quân Nam Hán.\n*   Nó khẳng định ý chí **quyết tâm giữ gìn non sông** của dân tộc ta.\n*   Ngô Quyền trở thành vị vua, mở ra một thời kỳ độc lập mới cho đất nước.\n\n_Nhờ có những người anh hùng như Ngô Quyền và chiến thắng Bạch Đằng, Việt Nam chúng ta mới được tự do và phát triển cho đến ngày nay!_'
        # Split the outline into meaningful chunks (words and punctuation)
        chunks = re.findall(r"\S+|\s+", outline)

        mock_usage = TokenUsage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            model="mock",
            provider="mock",
        )
        self.last_token_usage = mock_usage

        return chunks, mock_usage

    def make_outline_mock(
        self, outlineGenerateRequest: OutlineGenerateRequest
    ) -> Tuple[str, TokenUsage]:
        """Generate mock outline for testing purposes.
        Returns:
            Tuple: (outline, token_usage) - mock outline content and zero token usage.
        """

        sys_msg = self._system(
            "outline.system",
            outlineGenerateRequest.to_dict(),
        )
        print("System Prompt:", sys_msg)  # Debug print

        usr_sys_msg = self._system(
            "outline.user",
            outlineGenerateRequest.to_dict(),
        )
        print("User Prompt:", usr_sys_msg)  # Debug print

        outline = '### Giới thiệu: Một Cuộc Phiêu Lưu Lịch Sử Về Sông Bạch Đằng!\n\nChào các bạn nhỏ! Hôm nay, chúng ta sẽ cùng nhau du hành về quá khứ, đến với một khúc sông thật đặc biệt, nơi đã diễn ra một trận chiến lừng lẫy, giúp bảo vệ đất nước Việt Nam của chúng ta. Các bạn đã sẵn sàng chưa nào?\n\n*   Chúng ta sẽ khám phá câu chuyện về **Sông Bạch Đằng** – một dòng sông hùng vĩ.\n*   Tìm hiểu về những người anh hùng dũng cảm đã chiến đấu trên dòng sông này.\n*   Và hiểu tại sao trận chiến này lại quan trọng đến vậy!\n\n_Hãy chuẩn bị tinh thần để trở thành những nhà thám hiểm lịch sử nhé!_\n\n### Ai Đã Xâm Lược Nước Ta?\n\nNgày xưa, có những đội quân từ phương Bắc muốn xâm chiếm đất nước ta. Họ rất đông và mạnh mẽ, giống như một cơn bão sắp ập đến vậy.\n\n*   Quân địch đến từ **nước Nam Hán** (nay thuộc Trung Quốc).\n*   Họ muốn chiếm đóng và cai trị đất nước của chúng ta.\n*   Nhân dân ta rất lo sợ, nhưng không hề muốn bị mất nước.\n\n> Tưởng tượng xem, nếu có ai đó muốn lấy đi đồ chơi yêu thích của bạn, bạn sẽ làm gì? Ông cha ta cũng đã rất quyết tâm bảo vệ đất nước mình!\n\n### "Bẫy" Trên Sông: Ý Tưởng Tuyệt Vời Của Ngô Quyền!\n\nĐể chống lại quân địch mạnh mẽ, Ngô Quyền – vị tướng tài ba của chúng ta – đã nghĩ ra một kế hoạch vô cùng thông minh và độc đáo. Đó là sử dụng chính dòng sông Bạch Đằng để làm "chiến trường"!\n\n*   Ngô Quyền cho quân lính **cắm cọc nhọn** xuống lòng sông, ẩn dưới mặt nước lúc triều lên.\n*   Khi **triều rút**, những chiếc cọc này sẽ nhô lên, sẵn sàng đâm thủng thuyền địch.\n*   Đây là một cái bẫy thiên nhiên tuyệt vời!\n\n_Giống như chúng ta giăng bẫy chuột vậy đó, nhưng là bẫy cho thuyền lớn!_\n\n### Trận Chiến Rực Lửa Trên Sông!\n\nKhi quân Nam Hán hùng hổ tiến vào sông Bạch Đằng, họ đã mắc bẫy của Ngô Quyền.\n\n*   Thuyền địch bị **đâm thủng** bởi những chiếc cọc nhọn khi nước rút.\n*   Quân ta từ hai bên bờ sông đã **tấn công dữ dội**.\n*   Trận chiến diễn ra vô cùng ác liệt, nhưng quân ta đã chiến thắng vang dội!\n\n> Tiếng reo hò vang vọng khắp sông, đánh dấu một chiến thắng vẻ vang cho dân tộc!\n\n### Ý Nghĩa Lịch Sử: Vì Sao Chúng Ta Nhớ Mãi?\n\nChiến thắng sông Bạch Đằng không chỉ là một trận đánh hay, mà nó còn mang một ý nghĩa vô cùng to lớn đối với lịch sử Việt Nam.\n\n*   Trận chiến này đã giúp **giải phóng đất nước** khỏi ách đô hộ của quân Nam Hán.\n*   Nó khẳng định ý chí **quyết tâm giữ gìn non sông** của dân tộc ta.\n*   Ngô Quyền trở thành vị vua, mở ra một thời kỳ độc lập mới cho đất nước.\n\n_Nhờ có những người anh hùng như Ngô Quyền và chiến thắng Bạch Đằng, Việt Nam chúng ta mới được tự do và phát triển cho đến ngày nay!_'

        mock_usage = TokenUsage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            model="mock",
            provider="mock",
        )
        self.last_token_usage = mock_usage

        return outline, mock_usage

    def generate_image(self, request: ImageGenerateRequest):
        """Generate image based on text description"""

        usr_msg = self._system("image.user", {"prompt": request.prompt})

        result = self.llm_executor.generate_image(
            provider=request.provider,
            model=request.model,
            message=usr_msg,
            number_of_images=request.number_of_images,
            aspect_ratio=request.aspect_ratio,
            safety_filter_level=request.safety_filter_level,
            person_generation=request.person_generation,
            seed=request.seed,
            negative_prompt=request.negative_prompt,
        )
        return result

    def generate_image_mock(self, request: ImageGenerateRequest):
        """Generate mock image data for testing purposes."""
        sleep(random.uniform(0.3, 1.5))  # Simulate some processing delay
        with open("app/services/image_mock.png", "rb") as f:
            mock_image_data = base64.b64encode(f.read()).decode("utf-8")

        images = [mock_image_data for _ in range(request.number_of_images)]
        return {
            "images": images,
            "count": request.number_of_images,
            "error": None,
        }

    def generate_mindmap(self, request: MindmapGenerateRequest):
        sys_msg = self._system(
            "mindmap.system",
            request.to_dict(),
        )

        usr_msg = self._system(
            "mindmap.user",
            request.to_dict(),
        )

        result, token_usage = self.llm_executor.batch(
            provider=request.provider,
            model=request.model,
            messages=[
                SystemMessage(content=sys_msg),
                HumanMessage(content=usr_msg),
            ],
        )

        # Store token usage for later access
        self.last_token_usage = token_usage
        return result

    def generate_mindmap_mock(
        self, request: MindmapGenerateRequest
    ) -> Tuple[str, TokenUsage]:
        """Generate mock mindmap data for testing purposes.
        Returns:
            Tuple: (mindmap, token_usage) - mock mindmap content and zero token usage.
        """
        sleep(random.uniform(1.4, 1.5))  # Simulate some processing delay

        mock_mindmap = """
        ```json
    {
        "content": "Thế giới xung quanh em",
        "children": [
            {
            "content": "Động vật 🐾",
            "children": [
                {
                "content": "Động vật có vú",
                "children": [
                    { "content": "Chó - bạn thân của con người" },
                    { "content": "Mèo - loài vật tinh nghịch" },
                    { "content": "Voi - loài vật to lớn" }
                ]
                },
                {
                "content": "Chim 🐦",
                "children": [
                    { "content": "Chim sẻ - hót líu lo mỗi sáng" },
                    { "content": "Chim cánh cụt - sống ở xứ lạnh" },
                    { "content": "Đại bàng - chúa tể bầu trời" }
                ]
                },
                {
                "content": "Côn trùng 🐞",
                "children": [
                    { "content": "Ong - chăm chỉ làm mật" },
                    { "content": "Bướm - xinh đẹp với đôi cánh" }
                ]
                }
            ]
            },
            {
            "content": "Thực vật 🌱",
            "children": [
                {
                "content": "Cây xanh",
                "children": [
                    { "content": "Cây ăn quả - cho ta trái ngon" },
                    { "content": "Cây bóng mát - che rợp đường đi" }
                ]
                },
                {
                "content": "Hoa 🌸",
                "children": [
                    { "content": "Hoa hồng - biểu tượng của tình yêu" },
                    { "content": "Hoa hướng dương - luôn hướng về mặt trời" }
                ]
                },
                {
                "content": "Rau củ 🥕",
                "children": [
                    { "content": "Cà rốt - tốt cho mắt" },
                    { "content": "Bắp cải - nhiều vitamin" }
                ]
                }
            ]
            },
            {
            "content": "Thiên nhiên 🏞️",
            "children": [
                {
                "content": "Núi non hùng vĩ",
                "children": [
                    { "content": "Đỉnh núi cao vút" },
                    { "content": "Thung lũng xanh mướt" }
                ]
                },
                {
                "content": "Biển cả bao la 🌊",
                "children": [
                    { "content": "Sóng vỗ rì rào" },
                    { "content": "Sinh vật biển đa dạng" }
                ]
                },
                {
                "content": "Thời tiết ☀️🌧️",
                "children": [
                    { "content": "Trời nắng - ấm áp" },
                    { "content": "Trời mưa - mát mẻ" },
                    { "content": "Trời gió - thổi mạnh" }
                ]
                }
            ]
            }
        ]
    }
        ```
        """

        mock_usage = TokenUsage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            model="mock",
            provider="mock",
        )
        self.last_token_usage = mock_usage

        return mock_mindmap, mock_usage

    # ============ RAG ============
    def make_outline_with_rag(self, request: OutlineGenerateRequest):
        """Generate outline using LLM and save result.
        Args:
            request (OutlineGenerateRequest): Request object containing parameters for outline generation.
        Returns:
            Dict: A dictionary containing the generated outline.
        """
        sys_msg = self._system(
            "outline.system.rag",
            None,
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
        return result

    def make_presentation_with_rag(self, request: PresentationGenerateRequest):
        sys_msg = self._system(
            "presentation.system.rag",
            None,
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
        return result

    def generate_mindmap_with_rag(self, request: MindmapGenerateRequest):
        sys_msg = self._system(
            "mindmap.system.rag",
            None,
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
        return result
