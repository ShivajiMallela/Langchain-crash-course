from langchain_openai import ChatOpenAI
from langchain.prompts import  ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from config import set_environment
set_environment()

# create a chatopenai model
model = ChatOpenAI(model='gpt-4o')

# Create a prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system','You are a comedian who tells jokes about {topic}.'),
        ('human','Tell me {joke_count} jokes.')
    ]
)

# Create individual runnables (steps in the chain)
fromat_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages())) # type: ignore
parse_output = RunnableLambda(lambda x: x.content) # type: ignore

# Create the RunnableSequences (equivalent to the lcel chain)
chain = RunnableSequence(first=fromat_prompt, middle=[invoke_model], last=parse_output)

# run the chain
response = chain.invoke({"topic":"animals", 
                         "joke_count":3})

print(response)
