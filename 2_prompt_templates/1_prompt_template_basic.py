# Prompt Template Docs:
#   https://python.langchain.com/v0.2/docs/concepts/#prompt-templateshttps://python.langchain.com/v0.2/docs/concepts/#prompt-templates

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# Part1: Creating a chat prompt template using template string
# template = "Tell me a joke about {topic}"
# prompt_template = ChatPromptTemplate.from_template(template)

# print(f"---------Prompt from template--------")
# prompt = prompt_template.invoke({'topic':'cats'})
# print(prompt)

# Part2: Creating a prompt with multiple placeholders
# template_multiple = """
# You're an Helpful AI Assistant.
# Human: = Tell me a {adjective} story about a {animal}.
# Assistant:
# """

# prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
# prompt = prompt_multiple.invoke({'adjective':'funny',
#                                  'animal':'panda'})
# print(f"\n--------Prompt with Multiple placeholders:-------\n")
# print(prompt)

# part3: Prompt with System and Human messages (Using Tuples)
# messages = [
#     ("system", "You're a comedian who tells jokes about {topic}."),
#     ("human", "Tell me a {joke_count} jokes")
# ]

# prompt_template = ChatPromptTemplate.from_messages(messages=messages)
# prompt = prompt_template.invoke({"topic": "lawyers",
#                                  "joke_count":3})

# print(f"\n---- Prompt with System and Human Messages (Tuple)-----\n")
# print(prompt)

# To be note:
# This does work:
# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}."),
#     HumanMessage(content="Tell me 3 jokes."),
# ]
# prompt_template = ChatPromptTemplate.from_messages(messages)
# prompt = prompt_template.invoke({"topic": "lawyers"})
# print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
# print(prompt)


# This does NOT work:
# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}."),
#     HumanMessage(content="Tell me {joke_count} jokes."),
# ]
# prompt_template = ChatPromptTemplate.from_messages(messages)
# prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
# print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
# print(prompt)
