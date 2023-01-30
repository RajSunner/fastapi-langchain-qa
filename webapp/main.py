import os
import base64
import json
from typing import Union, Any
from os.path import dirname, abspath, join
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .chain import create_chain
from .chain import create_qa_chain
from .process import get_docs

app = FastAPI()

class Query(BaseModel):
    question: str
    history: Any
    key: str

@app.post("/llm")
async def query(request: Request, query: Query):
    # key = request.headers.get('X-Api-Key')
    key = query.key
    docs = get_docs(query.question)
    chain = create_qa_chain(key)
    answer = chain.run(input_documents=docs, question=query.question)
    history = json.dumps(query.history)
    results = {"answer": answer, "history": history}
    return results
