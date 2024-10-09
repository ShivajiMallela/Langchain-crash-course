from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from config import set_environment
set_environment()

chat_history = []

# set an initial system message 
system_message = SystemMessage(content="You're an Helpful AI assistant.")
chat_history.append(system_message)

model = ChatOpenAI(model='gpt-4o')

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    chat_history.append(HumanMessage(content=user_input))

    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))

    print(f'AI: {response}')

print(f'_____Message History_______')
print(chat_history)