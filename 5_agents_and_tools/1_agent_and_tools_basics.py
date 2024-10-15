from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

from config import set_environment
set_environment()

def get_current_time(*args, **kwargs):
    import datetime

    now = datetime.datetime.now() # Get current time
    return now.strftime("%I:%M %p")


tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful when you need to know the current time"
    )
]

# Pull the prompt template from the hub
# ReAct = Reason and Action
# https://smith.langchain.com/hub/hwchase17/react
prompt = hub.pull("hwchase17/react")

llm = ChatOpenAI(model='gpt-4o', temperature=0)

# Create the ReAct agent using the create_react_agent function
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True
)

# Create an Agent executer from the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, # type: ignore
    tools=tools,
    verbose=True
)

response = agent_executor.invoke({'input':'What time is it?'})

print("response: ", response)