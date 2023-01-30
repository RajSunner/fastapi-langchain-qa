from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import os


def create_chain(key):
    os.environ["OPENAI_API_KEY"] = key
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    os.environ["OPENAI_API_KEY"] = ''
    return chain

def create_qa_chain(key):
    os.environ["OPENAI_API_KEY"] = key
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    os.environ["OPENAI_API_KEY"] = ''
    return chain
