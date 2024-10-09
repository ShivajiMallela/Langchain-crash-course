from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

from config import set_environment
set_environment()

model = ChatOpenAI(model='gpt-4o')

# Define prompt templates (no need for seperate Runnable chains)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a comedian who tells jokes about {topic}."),
        ('human',"Tell me {joke_count} jokes.")
    ]
)

# Create the combined chain using Langchain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser()

# run the chain 
result = chain.invoke({'topic':"animals",
                       'joke_count':3})

print(result)