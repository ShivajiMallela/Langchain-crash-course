import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from config import set_environment
set_environment()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_dir = os.path.join(current_dir, 'db', 'chroma_db')
                              
if not os.path.exists(persistent_dir):
    print("Persistant directory does not exist. Initializing the vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path")
    
    # read the contents of the file
    loader = TextLoader(file_path=file_path)
    documents = loader.load()

    # split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents=documents)

    print("\n--Document chuck information----")
    print(f"Number of chunks: {len(docs)}")
    print(f"Sample chunk: \n{docs[0].page_content}")

    # Display info about split docs
    print("\n--Creating Embeddings--\n")
    embeddings = OpenAIEmbeddings(model='text_embedding-3-small')
    print(f"\n--Finished creating Embeddings--")

    print(f"\n--Creating vector store---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_dir)
    print(f"\n--finished Creating vector store---")

else:
    print(f"Vector store already exists, No need to initialize.")