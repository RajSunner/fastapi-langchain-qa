from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import pickle
from langchain.embeddings import HuggingFaceEmbeddings


def get_docs(query):
    with open("webapp/data/report-q.pkl", 'rb') as f:
        new_docsearch = pickle.load(f)
    docs = new_docsearch.similarity_search(query)
    return docs
