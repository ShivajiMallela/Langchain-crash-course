from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableBranch
from langchain.schema.output_parser import StrOutputParser

# load env variables
from config import set_environment
set_environment()

model = ChatOpenAI(model='gpt-4o')

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system','You are a Helpful AI Assistant.'),
        ('human','Generate a Thank you note for this positive feedback: {feedback}.')
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system','You are a Helpful AI Assistant.'),
        ('human','Generate a response addressing this negative feedback: {feedback}.')
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system','You are a Helpful AI Assistant.'),
        ('human','Generate a request for more details for this neutral feedback: {feedback}.')
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system','You are a Helpful AI Assistant.'),
        ('human','Generate a message to escalate this feedback to a human agent: {feedback}.')
    ]
)

# Define the feedback classification template
classification_template = ChatPromptTemplate.from_messages(
    [
        ('system','You are a Helpful AI Assistant.'),
        ('human','classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}')
    ]
)

# Define branches
branches = RunnableBranch(

    (
        lambda x: "positive" in x, # type: ignore
        positive_feedback_template | model | StrOutputParser()
    ),

    (
        lambda x: "negative" in x, # type: ignore
        negative_feedback_template | model | StrOutputParser()
    ),

    (
        lambda x: "neutral" in x, # type: ignore
        neutral_feedback_template | model | StrOutputParser()
    ),

    escalate_feedback_template | model | StrOutputParser()
)

# Create the classification chain
classification_chain = classification_template | model | StrOutputParser()

chain = classification_chain | branches

# Run the chain with an example review
# Good review - "The product is excellent. I really enjoyed using it and found it very helpful."
# Bad review - "The product is terrible. It broke after just one use and the quality is very poor."
# Neutral review - "The product is okay. It works as expected but nothing exceptional."
# Default - "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

review = "I'm not sure about the product yet. Can you tell me more about its features and benefits?"
result = chain.invoke({'feedback':review})

print(result)