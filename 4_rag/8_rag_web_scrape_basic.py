import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from config import set_environment
set_environment()

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistant_dir = os.path.join(db_dir, "chroma_db_dir")

urls = ["https://www.apple.com/"]

loader = WebBaseLoader(urls)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap = 0)
docs = text_splitter.split_documents(documents)

print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk: \n{docs[0].page_content}\n")

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

if not os.path.exists(persistant_dir):
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistant_dir)
    print(f"Vector store created..")
else:
    print(f"Vector store already created. No need to initialize it.")
    db = Chroma(persist_directory=persistant_dir, embedding_function=embeddings)

retriever = db.as_retriever(
    search_type = "similarity",
    search_kwargs = {'k':3}
)

query = "What new products are announced on Apple.com?"

relevant_docs = retriever.invoke(query)

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}: \n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'unknown')}\n")