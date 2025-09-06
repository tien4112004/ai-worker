from app.schemas.slide_content import OutlineGenerateRequest
from app.llms.service import LLMService
from app.repositories.llm_result_repository import llm_result_repository


class ContentService:
    def __init__(self, model_name: str, llm_service: LLMService):
        self.model_name = model_name
        self.llm_service = llm_service

    def make_slide(self, topic: str):
        """Generate slide content using LLM and save result."""
        prompt = f"Generate slide content for topic: {topic}"
        result = self.llm_service.generate_slide_content(topic)
        
        # Save to repository
        # result_id = llm_result_repository.save_result(
        #     prompt=prompt,
        #     response=result,
        #     model=self.model_name,
        #     metadata={"type": "slide", "topic": topic}
        # )
        
        return result

    def make_outline(self, request: OutlineGenerateRequest):
        """Generate outline using LLM with full request parameters and save result."""
        prompt = f"Generate outline for topic: {request.topic}"
        metadata = {
            "type": "outline",
            "topic": request.topic,
            "slide_count": request.slide_count,
            "learning_objective": request.learning_objective,
            "target_age": request.targetAge,
            "language": request.language
        }
        
        if hasattr(request, 'slide_count') and hasattr(request, 'learning_objective') and hasattr(request, 'targetAge'):
            # Generate detailed slide content if all parameters are available
            # result = self.llm_service.generate_slide_content(
            #     topic=request.topic,
            #     slide_count=request.slide_count,
            #     learning_objective=request.learning_objective,
            #     target_age=request.targetAge
            # )
            result = """I. Introduction to {request.topic}
A. Definition and scope
B. Importance and relevance 
II. Main Concepts
A. Key principles
B. Core components
III. Practical Applications
A. Real-world examples
B. Case studies
IV. Conclusion
A. Summary of key points
B. Future considerations
            """
            prompt = f"Generate {request.slide_count} slides for topic: {request.topic} with language is {request.language}, learning objective: {request.learning_objective}, target age: {request.targetAge}"
        else:
            # Fall back to simple outline
            # result = self.llm_service.generate_outline(request.topic)
            result = """I. Introduction to {request.topic}
A. Definition and scope
B. Importance and relevance 
II. Main Concepts
A. Key principles
B. Core components
III. Practical Applications
A. Real-world examples
B. Case studies
IV. Conclusion
A. Summary of key points
B. Future considerations
            """

        # Save to repository
        # result_id = llm_result_repository.save_result(
        #     prompt=prompt,
        #     response=result,
        #     model=self.model_name,
        #     metadata=metadata
        # )
        
        return result