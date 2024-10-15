from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

from config import set_environment
set_environment()

def get_current_time(*args, **kwargs):
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

def search_wiki(query):
    from wikipedia import summary

    try:
        return summary(query, sentences=2)
    except:
        print("I cannot find the info on that.")


tools = [
    Tool(
        name='Time',
        func=get_current_time,
        description="Useful when you need to know the current time."
    ),
    Tool(
        name='Wikipedia',
        func=search_wiki,
        description="Useful when you need to know information about a topic."
    )
]

prompt = hub.pull('hwchase17/structured-chat-agent')

# llm = ChatOpenAI(model='gpt-4o')

llm = Ollama(model='mistral')

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, # type: ignore
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors = True
)


initial_message = """You're an AI Assistant that can provide helpful answers using available tools.
                    If you are unable to answer, you can use the following tools: Time and Wikipedia."""

memory.chat_memory.add_message(SystemMessage(content=initial_message))

while True:
    user_input = input("User: ")
    if user_input.lower() == 'exit':
        break

    memory.chat_memory.add_message(HumanMessage(content=user_input))

    response = agent_executor.invoke({"input": user_input})
    print("Bot: ",response['output'])

    memory.chat_memory.add_message(AIMessage(content=response['output']))