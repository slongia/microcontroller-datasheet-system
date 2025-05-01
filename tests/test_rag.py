import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from rag_pipeline import RAGPipeline



def test_rag_pipeline():
    rag = RAGPipeline()
    result = rag.query("What is the clock speed of STM32F303?")
    assert isinstance(result, dict)
    assert "answer" in result
    assert "source" in result
