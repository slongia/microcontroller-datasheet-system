import pytest
from src.rag_pipeline import RAGPipeline


def test_rag_pipeline():
    rag = RAGPipeline()
    result = rag.query("What is the clock speed of STM32F303?")
    assert isinstance(result, dict)
    assert "answer" in result
    assert "source" in result
