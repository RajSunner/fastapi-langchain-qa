import os
import base64
import json
from typing import Union, Any
from os.path import dirname, abspath, join
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from webapp.chain import create_chain

current_dir = dirname(abspath(__file__))
static_path = join(current_dir, "static")

app = FastAPI()
app.mount("/ui", StaticFiles(directory=static_path), name="ui")


class Query(BaseModel):
    question: str
    history: Any


class Body(BaseModel):
    length: Union[int, None] = 20


@app.get('/')
def root():
    html_path = join(static_path, "index.html")
    return FileResponse(html_path)


@app.post('/generate')
def generate(body: Body):
    """
    Generate a pseudo-random token ID of twenty characters by default. Example POST request body:

    {
        "length": 20
    }
    """
    string = base64.b64encode(os.urandom(64))[:body.length].decode('utf-8')
    return {'token': string}


@app.post("/llm")
async def query(request: Request, query: Query):
    key = request.headers.get('X-Api-Key')

    chain = create_chain(key)
    answer = chain.run(query)
    history = json.dumps(query.history)
    results = {"answer": answer, "history": history}
    return results
