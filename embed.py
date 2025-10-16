from langchain_openai import OpenAIEmbeddings
import streamlit as st
import os
OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
def initialize_embeddings(openai_api_key=OPENAI_API_KEY):
    """
    Initialize embeddings using OpenAI or HuggingFace based on the availability of the OpenAI API key.

    Parameters:
        openai_api_key (str, optional): Your OpenAI API key. If not provided, it checks the environment variable.

    Returns:
        Embeddings object: An instance of OpenAIEmbeddings or HuggingFaceEmbeddings.
    """
    # Retrieve the OpenAI API key (default to an environment variable if not explicitly provided)
    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    if openai_api_key:  # Use OpenAI embeddings if the API key is available
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", 
             dimensions=512, # Use the desired OpenAI model
            openai_api_key=openai_api_key
        )
        print("Using OpenAIEmbeddings")
    return embeddings