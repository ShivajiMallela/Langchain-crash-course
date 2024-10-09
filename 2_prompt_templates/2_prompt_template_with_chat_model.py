from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import set_environment
set_environment()

model = ChatOpenAI(model='gpt-4o')

# # Part1: Create a ChatPromptTemplate using a template string
# print(f'===Prompt from template==')
# template = "Tell me a joke about {topic}."
# prompt_template = ChatPromptTemplate.from_template(template=template)

# prompt = prompt_template.invoke({'topic':'cats'})
# result = model.invoke(prompt)
# print(result.content)

# Part2: Prompt with Multiple placeholders
print(f'====Prompt with multiple placeholders====')
template_multiple = """
You're an helpful AI Assistant.
Human: Tell me a {adjective} short story about a {animal}.
Assistant: 
"""

prompt_multiple = ChatPromptTemplate.from_template(template=template_multiple)
prompt = prompt_multiple.invoke({'adjective':'funny',
                                 "animal":'elephant'})

result = model.invoke(prompt)
print(result.content)

# Part3: Prompt from messages
print(f"=====Prompt with system and human messages====")
messages = [
    ('system', "You're a comedian who tells jokes about {topic}."),
    ('human',"Tell me {joke_count} jokes")
]

prompt_template = ChatPromptTemplate.from_messages(messages=messages)
prompt = prompt_template.invoke({"topic":"heros", "joke_count": 3})
result = model.invoke(prompt)
print(result.content)