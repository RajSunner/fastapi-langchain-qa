import os
import time
import urllib.request
import PyPDF2
import io
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import pickle
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.weaviate import Weaviate
import weaviate

# def get_docs(query):
#     with open("webapp/data/report-q.pkl", 'rb') as f:
#         new_docsearch = pickle.load(f)
#     docs = new_docsearch.similarity_search(query)
#     return docs

def create_vectors_from_url(url, query, settings):
    URL = url
    req = urllib.request.Request(URL)
    remote_file = urllib.request.urlopen(req).read()
    remote_file_bytes = io.BytesIO(remote_file)
    reader = PyPDF2.PdfReader(remote_file_bytes)

    report_text = ''

    for x in range(len(reader.pages)):
      page = reader.pages[x]
      report_text += page.extract_text()

    report_splitter = CharacterTextSplitter(separator=" ",chunk_size=1000, chunk_overlap=100)
    texts_w = report_splitter.split_text(report_text)

    creds = weaviate.auth.AuthClientPassword(
        username=settings.emailw, password=settings.passwordw)
    client = weaviate.Client(
        auth_client_secret=creds,
        url=settings.urlw,
        additional_headers={
            'X-OpenAI-Api-Key': settings.okey
        }
    )
    client.schema.delete_all()
    client.schema.get()
    schema = {
        "classes": [
            {
                "class": "Paragraph",
                "description": "A written paragraph",
                "vectorizer": "text2vec-openai",
                "moduleConfig": {
                    "text2vec-openai": {
            "model": "ada",
            "modelVersion": "002",
            "type": "text"
                    }
                },
                "properties": [
                    {
                        "dataType": ["text"],
                        "description": "The content of the paragraph",
                        "moduleConfig": {
                            "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False
                            }
                        },
                        "name": "content",
                    },
                ],
            },
        ]
    }
    def callback(b):
        time.sleep(
            5
        )
    
    client.batch(
        batch_size=1,
        dynamic=True,
        creation_time=5,
        timeout_retries=3,
        connection_error_retries=3,
        callback=callback,
    
    )

    client.schema.create(schema)

    with client.batch as batch:
        for text in texts_w:
            batch.add_data_object({"content": text}, "Paragraph")
    
    vectorstore = Weaviate(client, "Paragraph", "content")
 
    docs = vectorstore.similarity_search(query)

    return docs

def get_docs_w(query, settings):

    creds = weaviate.auth.AuthClientPassword(
        username=settings.emailw, password=settings.passwordw)
    client = weaviate.Client(
        auth_client_secret=creds,
        url=settings.urlw,
        additional_headers={
            'X-OpenAI-Api-Key': settings.okey
        }
    )
    vectorstore = Weaviate(client, "Paragraph", "content")
    query = "What went wrong?"
    docs = vectorstore.similarity_search(query)
    return docs
