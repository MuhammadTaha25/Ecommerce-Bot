from doc_loader import load_csv

def chunking_documents(chunk_size=1500, chunk_overlap=150):
    """
    Fetch and split content from a Wikipedia page into manageable chunks.

    Parameters:
        url (str): The URL of the Wikipedia page to fetch content from.
        class_name (str): The class name of the HTML element to target for extraction.
        chunk_size (int, optional): The maximum size of each chunk (default is 1500 characters).
        chunk_overlap (int, optional): The overlap between chunks for context continuity (default is 100 characters).

    Returns:
        list: A list of document chunks as dictionaries.
    """
    # Load content using load_wikipedia_content function from doc_loader.py

    docs=load_csv("text [MConverter.eu].csv")
    # Initialize the RecursiveCharacterTextSplitter
    # Split the documents into smaller chunks
    chunks = docs
    return chunks
