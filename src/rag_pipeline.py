# from langchain.llms import HuggingFacePipeline
# from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from indexer import DatasheetIndexer
from dotenv import load_dotenv

# from langchain.schema.runnable import RunnableSequence
import os

load_dotenv()
index_path = os.getenv("INDEX_PATH", "data/index.faiss")
metadata_path = os.getenv("METADATA_PATH", "data/metadata.pkl")


class RAGPipeline:
    def __init__(self, index_path=index_path, metadata_path=metadata_path):
        self.indexer = DatasheetIndexer()
        self.indexer.load_index(index_path, metadata_path)
        self.llm = self._setup_llm()
        self.chain = self._setup_chain()

    def _setup_llm(self):
        """Set up Hugging Face LLM."""
        pipe = pipeline(
            "text-generation", model="distilgpt2", max_new_tokens=100, truncation=True
        )
        # pipe = pipeline("question-answering", model="deepset/roberta-base-squad2")

        return HuggingFacePipeline(pipeline=pipe)

    def _setup_chain(self):
        """Set up LangChain prompt and chain using RunnableSequence."""
        template = """
        Context: {context}
        Question: {question}
        Answer the question based on the context provided. If the answer is not in the context, say so.
        Answer: """

        # template = """
        # Context: {context}
        # Question: {question}
        # Provide a concise and direct answer to the question based only on the context above. If the answer is not present, reply: 'The answer is not in the provided context.'
        # Answer:
        # """

        prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )
        return prompt | self.llm

    def query(self, question):
        """Run RAG pipeline: retrieve context and generate answer."""
        results = self.indexer.search(question, k=1)
        if not results:
            return "No relevant information found."
        (filename, text), _ = results[0]
        context = text[:1000]  # Truncate for LLM input size
        response = self.chain.invoke({"context": context, "question": question})
        return {"answer": response, "source": filename}


if __name__ == "__main__":
    rag = RAGPipeline()
    question = "What is the clock speed of STM32F303?"
    result = rag.query(question)
    print(f"Answer: {result['answer']}\nSource: {result['source']}")
