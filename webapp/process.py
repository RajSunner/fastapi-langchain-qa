import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import pickle
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.weaviate import Weaviate
import weaviate

def get_docs(query):
    with open("webapp/data/report-q.pkl", 'rb') as f:
        new_docsearch = pickle.load(f)
    docs = new_docsearch.similarity_search(query)
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
