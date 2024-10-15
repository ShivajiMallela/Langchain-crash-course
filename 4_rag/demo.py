"""
This code is a practicing code using Ollama models which can be downloaded in the local using Ollama
"""


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os

from config import set_environment
set_environment()



## prompt template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user's queries"),
        ("user","Question:{question}")
    ]
)

## Streamlit framework

st.title('Langchain demo with Ollama')
input_text=st.text_input("Search the topic you want")

# Ollama LLM
llm=Ollama(model="llama2")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))