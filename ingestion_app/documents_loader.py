import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from llama_parse import LlamaParse

from ingestion_app.metadata_parser import (
    extract_metadata_from_path,
    get_metadata_summary,
)


class DocumentLoader:
    """Handles loading documents from various file formats for ingestion."""

    # Mapping of file extensions to their corresponding loaders
    # PDF uses LlamaParse, others use LangChain loaders
    SUPPORTED_EXTENSIONS = {
        ".pdf": "llama_parse",
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".docx": Docx2txtLoader,
    }

    def __init__(
        self,
        encoding: str = "utf-8",
        pdf_language: str = "vi",
        parsing_instruction: Optional[str] = None,
        use_premium_mode: bool = False,  # Changed to False when using auto_mode
        use_auto_mode: bool = True,  # NEW: Enable auto_mode for heavy image-table books
        max_pdf_size_mb: int = 150,
        skip_llama_parse_on_error: bool = True,
    ):
        """
        Initialize the DocumentLoader.

        Args:
            encoding: Character encoding for text files (default: utf-8)
            pdf_language: Language for PDF OCR (default: vi - Vietnamese)
            parsing_instruction: Custom parsing instructions for PDFs
            use_premium_mode: Use LlamaParse premium mode (default: False, use auto_mode instead)
            use_auto_mode: Use auto_mode for optimal parsing of image-heavy books (default: True)
            max_pdf_size_mb: Maximum PDF size in MB for LlamaParse (default: 150)
            skip_llama_parse_on_error: Skip LlamaParse for a file if it fails (default: True)
        """
        self.encoding = encoding
        self.pdf_language = pdf_language
        self.use_premium_mode = use_premium_mode
        self.use_auto_mode = use_auto_mode
        self.max_pdf_size_mb = max_pdf_size_mb
        self.skip_llama_parse_on_error = skip_llama_parse_on_error
        self._failed_files = set()  # Track files that failed with LlamaParse

        # Enhanced parsing instruction for image-heavy educational books with tables
        if parsing_instruction is None:
            self.parsing_instruction = (
                f"This is a Vietnamese educational textbook that contains text, images, tables, and diagrams. "
                f"The content is in {pdf_language} language. "
                "\n\n"
                "CRITICAL REQUIREMENTS:\n"
                "1. TABLES - MUST NOT SKIP: Extract ALL tables completely with full content. "
                "Convert tables to clean markdown format with proper alignment. "
                "Preserve all data, numbers, formulas, and relationships in tables. "
                "For complex multi-level tables, maintain the hierarchical structure.\n"
                "2. TEXT CONTENT: Extract all text accurately, preserving Vietnamese diacritical marks and special characters. "
                "Maintain paragraph breaks, formatting, and text flow.\n"
                "3. IMAGES & DIAGRAMS: Describe visual content that contains important educational information "
                "(charts, diagrams, illustrations). Include context about what the image represents.\n"
                "4. STRUCTURE: Preserve document hierarchy including chapters, sections, subsections, "
                "numbered lists, bullet points, and exercises.\n"
                "5. MATHEMATICAL & SCIENTIFIC CONTENT: Accurately capture formulas, equations, units, "
                "and scientific notation. Preserve subscripts, superscripts, and special symbols.\n"
                "6. EDUCATIONAL ELEMENTS: Mark exercises, examples, summaries, and key terms clearly.\n"
            )
        else:
            self.parsing_instruction = parsing_instruction

        self._init_llama_parser()

    def _init_llama_parser(self):
        """Initialize LlamaParse parser with configuration optimized for educational books."""
        try:
            llama_config = {
                "result_type": "markdown",
                "verbose": True,
                "language": "vi",
                "api_key": os.getenv("LLAMA_CLOUD_API_KEY"),
                "system_prompt": self.parsing_instruction,
            }

            # Use auto_mode for heavy image-table books (recommended for educational content)
            if self.use_auto_mode:
                llama_config["auto_mode"] = True
                llama_config["auto_mode_trigger_on_image_in_page_pct"] = (
                    0.3  # Trigger if page is >30% images
                )
                # Enable aggressive table extraction to ensure tables are captured
                llama_config["table_structure_parsing"] = "advanced"
            else:
                # Fallback to premium_mode if auto_mode is disabled
                llama_config["premium_mode"] = self.use_premium_mode

            # Enable comprehensive table extraction
            llama_config["skip_diagonal_text"] = False
            llama_config["take_screenshot"] = (
                True  # Helps with image-heavy pages
            )

            self.llama_parser = LlamaParse(**llama_config)
        except Exception as e:
            print(f"Warning: Failed to initialize LlamaParse: {e}")
            self.llama_parser = None

    def load_from_directory(
        self, directory_path: str, recursive: bool = True
    ) -> List[Document]:
        """
        Load all supported documents from a directory.

        Args:
            directory_path: Path to the directory containing documents
            recursive: Whether to search subdirectories recursively

        Returns:
            List of loaded Document objects

        Raises:
            ValueError: If directory path doesn't exist
        """
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")

        documents = []
        pattern = "**/*" if recursive else "*"

        for file_path in dir_path.glob(pattern):
            if (
                file_path.is_file()
                and file_path.suffix in self.SUPPORTED_EXTENSIONS
            ):
                try:
                    docs = self.load_file(str(file_path))
                    documents.extend(docs)
                    print(f"✓ Loaded: {file_path.name} ({len(docs)} chunks)")
                except Exception as e:
                    print(f"✗ Failed to load {file_path.name}: {str(e)}")

        return documents

    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a single document file.

        Args:
            file_path: Path to the document file

        Returns:
            List of Document objects (may contain multiple pages/chunks)

        Raises:
            ValueError: If file doesn't exist or unsupported format
        """
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise ValueError(f"File not found: {file_path}")

        file_extension = path.suffix.lower()
        if file_extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS.keys())}"
            )

        # Handle PDF files with LlamaParse
        if file_extension == ".pdf":
            documents = self._load_pdf(file_path)
        else:
            # Handle other file types with LangChain loaders
            loader_class = self.SUPPORTED_EXTENSIONS[file_extension]

            # Special handling for TextLoader to specify encoding
            if loader_class == TextLoader:
                loader = loader_class(file_path, encoding=self.encoding)
            else:
                loader = loader_class(file_path)

            documents = loader.load()

        # Extract educational metadata from filename
        educational_metadata = extract_metadata_from_path(file_path)

        # Add file metadata and educational metadata to all documents
        for doc in documents:
            # Basic file metadata
            doc.metadata["source_file"] = path.name
            doc.metadata["file_path"] = str(path.absolute())
            doc.metadata["file_type"] = file_extension

            # Educational metadata (if available)
            if educational_metadata.get("has_metadata"):
                doc.metadata["grade"] = educational_metadata["grade"]
                doc.metadata["subject_code"] = educational_metadata[
                    "subject_code"
                ]
                doc.metadata["subject_name"] = educational_metadata[
                    "subject_name"
                ]
                doc.metadata["metadata_summary"] = get_metadata_summary(
                    educational_metadata
                )

        return documents

    def _load_pdf(self, file_path: str) -> List[Document]:
        """
        Load PDF file using LlamaParse for advanced parsing.

        Args:
            file_path: Path to PDF file

        Returns:
            List of Document objects
        """
        # Check if this file previously failed and should be skipped
        if self.skip_llama_parse_on_error and file_path in self._failed_files:
            print(
                f"⚠️  Skipping LlamaParse for {file_path} (previously failed)"
            )
            from langchain_community.document_loaders import PyMuPDFLoader

            loader = PyMuPDFLoader(file_path)
            return loader.load()

        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > self.max_pdf_size_mb:
            print(
                f"⚠️  PDF too large ({file_size_mb:.1f}MB > {self.max_pdf_size_mb}MB): {file_path}"
            )
            print(f"   Skipping LlamaParse and using basic loader.")
            print(
                f"   Tip: Increase max_pdf_size_mb or split the PDF into smaller files."
            )
            if self.skip_llama_parse_on_error:
                self._failed_files.add(file_path)
            from langchain_community.document_loaders import PyMuPDFLoader

            loader = PyMuPDFLoader(file_path)
            return loader.load()

        try:
            # Use LlamaParse for advanced parsing
            llama_documents = self.llama_parser.load_data(file_path)

            # Convert LlamaIndex documents to LangChain documents
            langchain_docs = []
            for idx, llama_doc in enumerate(llama_documents):
                langchain_doc = Document(
                    page_content=llama_doc.text,
                    metadata={
                        "page": idx + 1,
                        "source": file_path,
                        **llama_doc.metadata,
                    },
                )
                langchain_docs.append(langchain_doc)

            return langchain_docs

        except Exception as e:
            error_message = str(e).lower()

            # Track this file as failed
            if self.skip_llama_parse_on_error:
                self._failed_files.add(file_path)

            # Check for token limit exceeded errors
            if any(
                keyword in error_message
                for keyword in [
                    "token limit",
                    "tokens exceeded",
                    "context length",
                    "too many tokens",
                    "maximum context",
                    "rate limit",
                    "quota exceeded",
                    "too large",
                ]
            ):
                print(f"⚠️  Token/Rate limit exceeded for {file_path}")
                print(f"   Error: {e}")
                print(
                    f"   This PDF is too large for LlamaParse (exceeds token limit)."
                )
                print(f"   Recommendation:")
                print(f"   - Split the PDF into smaller files")
                print(f"   - Increase max_pdf_size_mb parameter")
                print(f"   - Use basic loader for this file")
                print(f"   Falling back to basic PDF loader...")
            else:
                print(f"❌ Error parsing PDF with LlamaParse: {e}")
                print(f"   Falling back to basic PDF loader...")

            # Fallback to basic PDF loader
            try:
                from langchain_community.document_loaders import PyMuPDFLoader

                loader = PyMuPDFLoader(file_path)
                return loader.load()
            except Exception as fallback_error:
                print(f"❌ Basic PDF loader also failed: {fallback_error}")
                print(f"   Skipping file: {file_path}")
                return []  # Return empty list if both loaders fail

    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.

        Returns:
            List of supported file extensions
        """
        return list(self.SUPPORTED_EXTENSIONS.keys())

    def get_failed_files(self) -> set:
        """
        Get list of files that failed with LlamaParse.

        Returns:
            Set of file paths that failed
        """
        return self._failed_files.copy()

    def clear_failed_files(self):
        """Clear the list of failed files."""
        self._failed_files.clear()

    def get_loading_stats(self) -> dict:
        """
        Get statistics about document loading.

        Returns:
            Dictionary with loading statistics
        """
        return {
            "failed_files_count": len(self._failed_files),
            "failed_files": list(self._failed_files),
            "max_pdf_size_mb": self.max_pdf_size_mb,
        }
