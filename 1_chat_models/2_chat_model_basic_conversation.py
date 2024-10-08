from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables
import os
from config import set_environment
set_environment()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# SystemMessage:
#   Message for priming AI behavior(To tell AI how to respond to user queries),
#    usually passed in as the first of a sequence of input messages.
# HumanMessage:
#   Message from a human to the AI model.
messages = [
    SystemMessage(content="Solve the given math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from AI: {result.content}")

# AIMessage:
#   Message from an AI.
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="What is 10 times 5?"),
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from AI: {result.content}")
