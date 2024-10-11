import os

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from config import set_environment
set_environment()

# Define the persistant directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, 'db')
persistant_directory = os.path.join(current_dir, 'db', 'chroma_db_with_metadata')

# define the embeddings
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# load the existing vector store with embedding function
db  = Chroma(persist_directory=persistant_directory,
             embedding_function=embeddings)

# Define the query
query = "How did juliet die?"

retreiver = db.as_retriever(
    search_type = 'similarity_score_threshold',
    search_kwargs = {'k':3, 'score_threshold':0.2}
)

relevant_docs = retreiver.invoke(query)

# Display the relevant docs to the query
print("Relevant Documents ⬇️")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document: \n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")