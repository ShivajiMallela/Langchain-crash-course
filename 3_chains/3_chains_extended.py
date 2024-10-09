from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda
from config import set_environment
set_environment()

model = ChatOpenAI(model='gpt-4o')

# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# define additional processing steps using runnable lambda
uppercase_output = RunnableLambda(lambda x: x.upper()) # type: ignore
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}") # type: ignore

chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

result = chain.invoke({'topic':'animals',
                       'joke_count': 3})

print(result)