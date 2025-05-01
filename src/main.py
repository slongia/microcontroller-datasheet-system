from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import RAGPipeline

app = FastAPI(title="Microcontroller Datasheet Knowledge System")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    source: str


@app.post("/query", response_model=QueryResponse)
async def query_datasheet(request: QueryRequest):
    rag = RAGPipeline()
    result = rag.query(request.question)
    return QueryResponse(**result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
