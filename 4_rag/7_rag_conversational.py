import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st

from config import set_environment
set_environment()

# Define the persistant directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistant_dir = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Define the embedding model
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Load the existing vector store with an embedding function
db = Chroma(persist_directory=persistant_dir, embedding_function=embeddings)

# creating a retreiver for querying the vector store
retriever = db.as_retriever(
    search_type = 'similarity',
    search_kwargs = {'k':3}
)

# create a ChatOpenAI model
llm = ChatOpenAI(model='gpt-4o')

# contextualize question prompt
# This prompt helps AI understand that it has to make a standalone question (that is independent of other information) for the user's query
# which means it rephrases the prompt given by based on the chat history to be a standalone question

Standalone_question_prompt = (
    """
    Given a chat history and the latest user question which
    might reference context in the chat history, formulate 
    a standalone question which can be understood without
    the chat history. Do Not answer the question, just 
    reformulate it if Needed. and otherwise return it as it is.
    """
)

Standalone_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', Standalone_question_prompt),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ]
)

# Create a history-aware retreiver
# This uses the LLM to help reformulate the question based on the chat history
history_aware_retriever = create_history_aware_retriever(llm, retriever, Standalone_prompt)


# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown

qa_system_prompt = (
    """
    You are an assistant for question-answering tasks.
    use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise. \n\n {context}
    """
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ('human', '{input}')
    ]
)

# create a chain to combine docs for answering question
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Function to simulate a continual chat
def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        result = rag_chain.invoke({'input':query, 'chat_history': chat_history})
        print(f"AI: {result['answer']}")
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result['answer']))

# Main function to start the continual chat
if __name__=="__main__":
    continual_chat()