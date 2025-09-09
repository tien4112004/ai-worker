import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_outline_request():
    """Sample request data for outline generation."""
    return {
        "topic": "Introduction to Machine Learning",
        "model": "gemini-2.5-flash-lite",
        "language": "en",
        "slide_count": 5,
        "learning_objective": "Understand basic ML concepts",
        "targetAge": "18-25",
    }
