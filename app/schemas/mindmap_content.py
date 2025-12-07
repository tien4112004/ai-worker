from typing import List, Optional

from pydantic import BaseModel, Field
from typing_extensions import Literal


class MindmapGenerateRequest(BaseModel):
    topic: str
    language: str
    maxDepth: int = Field(
        default=4, description="Maximum depth of the mindmap"
    )
    maxBranchesPerNode: int = Field(
        default=3, description="Maximum branches per node"
    )
    provider: str
    model: str

    def to_dict(self):
        return {
            "topic": self.topic,
            "language": self.language,
            "maxDepth": self.maxDepth,
            "maxBranchesPerNode": self.maxBranchesPerNode,
        }
