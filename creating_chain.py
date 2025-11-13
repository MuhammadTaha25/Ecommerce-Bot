from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain.schema import StrOutputParser
from operator import itemgetter

OPENAI_API_KEY =st.secrets['OPENAI_API_KEY']
GEMINI_API_KEY = st.secrets['GEMINI_API_KEY'] 
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_expert_chain(LLM=None, retriever=None):
    """
    Create a chain for answering questions as an expert on Elon Musk.

    Parameters:
        llm (object): The language model to use for generating responses.
        retriever (object): A retriever for fetching relevant context based on the question.

    Returns:
        object: A configured chain for answering questions about Elon Musk.
    """
    # Define the prompt template
    prompt_str ="""You are a helpful assistant that answers user queries using the provided product catalog (Context).

Context: {context}
Question: {question}
Answer in detail, and if the answer is not contained within the context, say 'I am Trained to answer queries related to Ecommerce products only. Please rephrase your query to be related to E-commerce products.'
   """ 
    _prompt = ChatPromptTemplate.from_template(prompt_str)

    # Chain setup with history
    setup = {
        "question": itemgetter("question"),
        "context": itemgetter("question") | retriever | format_docs,
    }
    _chain = setup | _prompt | LLM | StrOutputParser()


    return _chain



