# Example Source: https://python.langchain.com/v0.2/docs/integrations/memory/google_firestore/

from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_openai import ChatOpenAI

"""
Steps to replicate this example:
1. Create a Firebase account
2. Create a new Firebase project
    - Copy the project ID
3. Create a Firestore database in the Firebase project
4. Install the Google Cloud CLI on your computer
    - https://cloud.google.com/sdk/docs/install
    - Authenticate the Google Cloud CLI with your Google account
        - https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
    - Set your default project to the new Firebase project you created
5. Enable the Firestore API in the Google Cloud Console:
    - https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=crewai-automation
"""

from config import set_environment
set_environment()

# set Firebase Firestore
PROJECT_ID = "langchain-tutorials-dabf9"
SESSION_ID = "session_1"
COLLECTION_NAME = "chat_history"

# initialize the Firestore client
print("Initializing the Firestore client..")
client = firestore.Client(project=PROJECT_ID)

#Initialize Firestore chat message history
print(f"Initializing Firestore chat message history....")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client
)

print(f'Chat History Initialized.')
print(f'current chat history:', chat_history.messages)

# Initialize the chat model
model = ChatOpenAI(model='gpt-4o')

print(f"Start chatting with the AI. Type 'exit' to quit.")

while True:
    user_input = input('You: ')
    if user_input.lower() == 'exit':
        break

    chat_history.add_user_message(user_input)

    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content) # type: ignore

    print(f'AI: {ai_response.content}')