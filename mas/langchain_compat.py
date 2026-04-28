"""Compatibility helpers for LangChain import path differences."""

try:
    from langchain.docstore.document import Document  # type: ignore
except ImportError:
    from langchain_core.documents import Document  # type: ignore

