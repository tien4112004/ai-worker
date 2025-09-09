"""Test prompt loading functionality."""

from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from app.prompts.loader import PromptSpec, PromptStore


class TestPromptStore:
    """Test PromptStore functionality."""

    @pytest.fixture
    def mock_registry_data(self):
        """Mock registry data."""
        return {
            "prompts": {
                "test.prompt": {"path": "test/prompt.st", "format": "st"},
                "test.with_defaults": {
                    "path": "test/with_defaults.st",
                    "format": "st",
                    "defaults": "test/defaults.yaml",
                },
            }
        }

    @pytest.fixture
    def prompt_store(self):
        """Create a PromptStore instance."""
        return PromptStore()

    def test_prompt_spec_creation(self):
        """Test PromptSpec creation."""
        spec = PromptSpec(
            key="test.key", path=Path("test/path.st"), format="st"
        )
        assert spec.key == "test.key"
        assert spec.path == Path("test/path.st")
        assert spec.format == "st"
        assert spec.defaults_path is None

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='prompts:\n  test.key:\n    path: "test.st"',
    )
    @patch("yaml.safe_load")
    def test_load_registry(
        self, mock_yaml, mock_file, prompt_store, mock_registry_data
    ):
        """Test registry loading."""
        mock_yaml.return_value = mock_registry_data

        # Clear the cache to ensure fresh load
        prompt_store._load_registry.cache_clear()

        result = prompt_store._load_registry()
        assert result == mock_registry_data

    @patch.object(PromptStore, "_load_registry")
    def test_spec_creation_from_registry(
        self, mock_load_registry, prompt_store, mock_registry_data
    ):
        """Test PromptSpec creation from registry."""
        mock_load_registry.return_value = mock_registry_data

        spec = prompt_store._spec("test.prompt")

        assert spec.key == "test.prompt"
        assert spec.path.name == "prompt.st"
        assert spec.format == "st"

    @patch.object(PromptStore, "_load_registry")
    def test_spec_not_found(self, mock_load_registry, prompt_store):
        """Test KeyError when prompt key not found."""
        mock_load_registry.return_value = {"prompts": {}}

        with pytest.raises(
            KeyError, match="Prompt key not found: nonexistent"
        ):
            prompt_store._spec("nonexistent")

    @patch.object(Path, "read_text")
    def test_load_text(self, mock_read_text, prompt_store):
        """Test text loading with caching."""
        mock_read_text.return_value = "Test prompt content"

        path = Path("test.st")
        result = prompt_store._load_text(path)

        assert result == "Test prompt content"
        mock_read_text.assert_called_once_with(encoding="utf-8")
