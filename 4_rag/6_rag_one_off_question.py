import os
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import set_environment
set_environment()

# Define the persistant directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistant_directory = os.path.join(current_dir, 'db', 'chroma_db_with_metadata')

# Define the embedding model
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Load the existing vector store with embedding function
db = Chroma(persist_directory=persistant_directory,
            embedding_function=embeddings)

# Define the user's question
query = "How can i learn more about langchain?"

# Retrieve relevant documents based on the query
retreiver = db.as_retriever(
    search_type='similarity',
    search_kwargs = {'k':2}
)
relevant_docs = retreiver.invoke(query)

# Display the relevant results with metadata
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# combine the query and the relevant document contents
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\n Please provide an answer based only on the provided contents. if the answer is not found in the documents, respond with 'Sorry, i'm not surea about that.'"
)

# create a ChatOpenAI model
model = ChatOpenAI(model='gpt-4o')

# define the messages for the model
messages = [
    SystemMessage(content="You're an helpful Assistant"),
    HumanMessage(content=combined_input)
]

# Invoke the model with combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n-- Generated Response ---")
print(result.content)