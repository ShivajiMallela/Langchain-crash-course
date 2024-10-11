"""
This code is to create vector store of multiple documents (just like "1a_rag_basics.py" which we created for one doc)
"""

import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from config import set_environment
set_environment()

# Define the directory containing the text files and the persistant directory
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistant_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books directory: {books_dir}")
print(f"Persistant directory: {persistant_directory}")

# check if the chroma vector store already exists
if not os.path.exists(persistant_directory):
    print(f"Persistant directory does not exist. Initializing vector store...")

    # Ensure the books directory exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )
    
    # List all text files in the directory 
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    # read the file content from each file and store it with metadata
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path=file_path)
        book_docs = loader.load()
        for doc in book_docs:
            # Add metadata to each document indicating its source
            doc.metadata = {'source': book_file}
            documents.append(doc)
    # split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap = 0)
    docs = text_splitter.split_documents(documents=documents)

    # Display information about the split documents
    print("\n--Document chunks information---\n")
    print(f"Number of document chunks: {len(docs)}")

    # create embeddings
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    print("--Embeddings created--")

    # create the vector store and persist it
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistant_directory)
    print(f"Vector store created")

else:
    print(f"Vector store already exists. No need to intialize.")