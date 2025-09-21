from typing import Any, Dict, Generator

from langchain_core.messages import HumanMessage, SystemMessage

from app.llms.executor import LLMExecutor
from app.prompts.loader import PromptStore
from app.repositories.llm_result_repository import llm_result_repository
from app.schemas.slide_content import (
    OutlineGenerateRequest,
    PresentationGenerateRequest,
)


class ContentService:
    def __init__(self, llm_executor: LLMExecutor, prompt_store: PromptStore):
        self.llm_executor = llm_executor or LLMExecutor()
        self.prompt_store = prompt_store or PromptStore()

    def _system(self, key: str, vars: Dict[str, Any] | None) -> str:
        return self.prompt_store.render(key, vars)

    # Presentation Generation
    def make_presentation_stream(self, request: PresentationGenerateRequest):
        """Generate slide content using LLM and save result.
        Args:
            request (PresentationGenerateRequest): Request object containing parameters for slide generation.
        Returns:
            Generator: A generator yielding parts of the generated slide content.
        """
        sys_msg = self._system(
            "presentation.system",
            None,
        )

        usr_msg = self._system(
            "presentation.user",
            request.to_dict(),
        )

        result = self.llm_executor.stream(
            provider=request.provider,
            model=request.model,
            messages=[
                SystemMessage(content=sys_msg),
                HumanMessage(content=usr_msg),
            ],
        )

        return result

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

        result = self.llm_executor.batch(
            provider=request.provider,
            model=request.model,
            messages=[
                SystemMessage(content=sys_msg),
                HumanMessage(content=usr_msg),
            ],
        )

        return result

    # Outline Generation
    def make_outline_stream(self, request: OutlineGenerateRequest):
        """Generate outline using LLM and save result.
        Args:
            request (OutlineGenerateRequest): Request object containing parameters for outline generation.
        Returns:
            Generator: A generator yielding parts of the generated outline.
        """
        sys_msg = self._system(
            "outline.system",
            None,
        )

        usr_msg = self._system(
            "outline.user",
            request.to_dict(),
        )

        result = self.llm_executor.stream(
            provider=request.provider,
            model=request.model,
            messages=[
                SystemMessage(content=sys_msg),
                HumanMessage(content=usr_msg),
            ],
        )

        return result

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

        result = self.llm_executor.batch(
            provider=request.provider,
            model=request.model,
            messages=[
                SystemMessage(content=sys_msg),
                HumanMessage(content=usr_msg),
            ],
        )

        return result

    def make_presentation_mock(self, request: PresentationGenerateRequest):
        """Generate mock slide content for testing purposes.
        Returns:
            Dict: A dictionary containing the mock slide content.
        """

        sys_msg = self._system(
            "presentation.system",
            request.to_dict(),
        )
        print("System Prompt:", sys_msg)  # Debug print

        result = '```json\n{\n  "slides": [\n    {\n      "type": "main_image",\n      "data": {\n        "image": "Children looking excitedly at an old map of Vietnam with a river highlighted",\n        "content": "Giới thiệu: Một Cuộc Phiêu Lưu Lịch Sử Về Sông Bạch Đằng!"\n      }\n    },\n    {\n      "type": "two_column_with_image",\n      "title": "Ai Đã Xâm Lược Nước Ta?",\n      "data": {\n        "items": [\n          "Quân địch đến từ nước Nam Hán.",\n          "Họ muốn chiếm đất nước ta.",\n          "Nhân dân ta không muốn bị mất nước."\n        ],\n        "image": "Illustration of ancient Chinese warships sailing towards Vietnamese shores"\n      }\n    },\n    {\n      "type": "two_column_with_image",\n      "title": "\\"Bẫy\\" Trên Sông: Ý Tưởng Của Ngô Quyền!",\n      "data": {\n        "items": [\n          "Ngô Quyền cho cắm cọc nhọn dưới sông.",\n          "Cọc ẩn dưới nước lúc triều lên.",\n          "Nhô lên đâm thủng thuyền địch khi nước rút."\n        ],\n        "image": "Illustration of a wooden stake hidden underwater in a river with a boat approaching"\n      }\n    },\n    {\n      "type": "two_column_with_image",\n      "title": "Trận Chiến Rực Lửa Trên Sông!",\n      "data": {\n        "items": [\n          "Thuyền địch mắc bẫy, bị đâm thủng.",\n          "Quân ta tấn công từ hai bên bờ.",\n          "Chiến thắng vang dội cho dân tộc!"\n        ],\n        "image": "Illustration of Vietnamese soldiers attacking enemy ships from the riverbanks during a battle"\n      }\n    },\n    {\n      "type": "main_image",\n      "data": {\n        "image": "Illustration of a proud Vietnamese flag waving over a peaceful landscape",\n        "content": "Chiến thắng Bạch Đằng giúp đất nước ta mãi mãi tự do!"\n      }\n    }\n  ]\n}\n```'
        return result

    def make_presentation_stream_mock(self):
        """Generate mock slide content stream for testing purposes.
        Yields:
            str: Parts of the mock slide content.
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

        import json
        import re
        import time

        # Create the complete JSON structure
        complete_json = {"slides": slides}

        # Convert to JSON string
        json_str = json.dumps(complete_json, ensure_ascii=False, indent=2)

        # Split into meaningful chunks (words, punctuation, and whitespace)
        chunks = re.findall(r"\S+|\s+", json_str)

        for chunk in chunks:
            # Add a small delay to simulate streaming
            time.sleep(0.01)
            yield chunk

    def make_outline_stream_mock(self) -> Generator[str, None, None]:
        """Generate mock outline stream for testing purposes.
        Yields:
            str: Parts of the mock outline.
        """
        import re
        import time

        outline = '### Giới thiệu: Một Cuộc Phiêu Lưu Lịch Sử Về Sông Bạch Đằng!\n\nChào các bạn nhỏ! Hôm nay, chúng ta sẽ cùng nhau du hành về quá khứ, đến với một khúc sông thật đặc biệt, nơi đã diễn ra một trận chiến lừng lẫy, giúp bảo vệ đất nước Việt Nam của chúng ta. Các bạn đã sẵn sàng chưa nào?\n\n*   Chúng ta sẽ khám phá câu chuyện về **Sông Bạch Đằng** – một dòng sông hùng vĩ.\n*   Tìm hiểu về những người anh hùng dũng cảm đã chiến đấu trên dòng sông này.\n*   Và hiểu tại sao trận chiến này lại quan trọng đến vậy!\n\n_Hãy chuẩn bị tinh thần để trở thành những nhà thám hiểm lịch sử nhé!_\n\n### Ai Đã Xâm Lược Nước Ta?\n\nNgày xưa, có những đội quân từ phương Bắc muốn xâm chiếm đất nước ta. Họ rất đông và mạnh mẽ, giống như một cơn bão sắp ập đến vậy.\n\n*   Quân địch đến từ **nước Nam Hán** (nay thuộc Trung Quốc).\n*   Họ muốn chiếm đóng và cai trị đất nước của chúng ta.\n*   Nhân dân ta rất lo sợ, nhưng không hề muốn bị mất nước.\n\n> Tưởng tượng xem, nếu có ai đó muốn lấy đi đồ chơi yêu thích của bạn, bạn sẽ làm gì? Ông cha ta cũng đã rất quyết tâm bảo vệ đất nước mình!\n\n### "Bẫy" Trên Sông: Ý Tưởng Tuyệt Vời Của Ngô Quyền!\n\nĐể chống lại quân địch mạnh mẽ, Ngô Quyền – vị tướng tài ba của chúng ta – đã nghĩ ra một kế hoạch vô cùng thông minh và độc đáo. Đó là sử dụng chính dòng sông Bạch Đằng để làm "chiến trường"!\n\n*   Ngô Quyền cho quân lính **cắm cọc nhọn** xuống lòng sông, ẩn dưới mặt nước lúc triều lên.\n*   Khi **triều rút**, những chiếc cọc này sẽ nhô lên, sẵn sàng đâm thủng thuyền địch.\n*   Đây là một cái bẫy thiên nhiên tuyệt vời!\n\n_Giống như chúng ta giăng bẫy chuột vậy đó, nhưng là bẫy cho thuyền lớn!_\n\n### Trận Chiến Rực Lửa Trên Sông!\n\nKhi quân Nam Hán hùng hổ tiến vào sông Bạch Đằng, họ đã mắc bẫy của Ngô Quyền.\n\n*   Thuyền địch bị **đâm thủng** bởi những chiếc cọc nhọn khi nước rút.\n*   Quân ta từ hai bên bờ sông đã **tấn công dữ dội**.\n*   Trận chiến diễn ra vô cùng ác liệt, nhưng quân ta đã chiến thắng vang dội!\n\n> Tiếng reo hò vang vọng khắp sông, đánh dấu một chiến thắng vẻ vang cho dân tộc!\n\n### Ý Nghĩa Lịch Sử: Vì Sao Chúng Ta Nhớ Mãi?\n\nChiến thắng sông Bạch Đằng không chỉ là một trận đánh hay, mà nó còn mang một ý nghĩa vô cùng to lớn đối với lịch sử Việt Nam.\n\n*   Trận chiến này đã giúp **giải phóng đất nước** khỏi ách đô hộ của quân Nam Hán.\n*   Nó khẳng định ý chí **quyết tâm giữ gìn non sông** của dân tộc ta.\n*   Ngô Quyền trở thành vị vua, mở ra một thời kỳ độc lập mới cho đất nước.\n\n_Nhờ có những người anh hùng như Ngô Quyền và chiến thắng Bạch Đằng, Việt Nam chúng ta mới được tự do và phát triển cho đến ngày nay!_'
        # Split the outline into meaningful chunks (words and punctuation)
        chunks = re.findall(r"\S+|\s+", outline)

        for chunk in chunks:
            yield chunk

    def make_outline_mock(
        self, outlineGenerateRequest: OutlineGenerateRequest
    ):
        """Generate mock outline for testing purposes.
        Returns:
            Dict: A dictionary containing the mock outline.
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
        return outline
