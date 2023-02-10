import json
from typing import Any
from functools import lru_cache
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
from .chain import create_chain
from .chain import create_qa_chain
from .process import get_docs_w
from .process import create_vectors_from_url
from .config import Settings

app = FastAPI()


@lru_cache()
def get_settings():
    return Settings()


class Query(BaseModel):
    question: str


class File(BaseModel):
    question: str
    file_url: str


@app.post("/llm")
async def query(request: Request, query: Query, settings: Settings = Depends(get_settings)):
    key = request.headers.get('X-Api-Key')
    docs = get_docs_w(query.question, settings)
    chain = create_qa_chain(key)
    answer = chain.run(input_documents=docs, question=query.question)
    # history = json.dumps(query.history)
    results = {"message": answer}
    return results


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/file")
async def file_vec(request: Request, file: File, settings: Settings = Depends(get_settings)):
    key = request.headers.get('X-Api-Key')
    docs = create_vectors_from_url(file.file_url, file.question, settings)
    chain = create_qa_chain(key)
    answer = chain.run(input_documents=docs, question=file.question)
    results = {"message": answer}
    return results
