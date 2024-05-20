import textwrap
from scripts.RAG_model import query_engine


def query(question: str):
    # return the answer based on the documents provided
    response = query_engine.query(question)

    text_response = response.response.split('\n\n')[0]
    text_response = textwrap.fill(text_response, width=120)
    return text_response
