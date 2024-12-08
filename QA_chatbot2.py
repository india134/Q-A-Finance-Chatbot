#!/usr/bin/env python
# coding: utf-8

# In[28]:





# In[ ]:


import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os


# In[12]:


from dotenv import load_dotenv


# In[13]:


load_dotenv()


# In[15]:


LANGCHAIN_API_KEY="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"


# In[18]:


os.environ["LANGCHAIN_API_KEY"] = "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"


# In[19]:


## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot With Ollama"


# In[22]:


#Prompttemplate
prompt=ChatPromptTemplate.from_messages(
[
    ("system","You are a highly skilled finance expert. Always respond with detailed and accurate financial advice based on user queries."),("user","Question:{question}")
])


# In[31]:


from langchain.schema import AIMessage, HumanMessage, SystemMessage

def generate_response(question, llm, temperature, max_tokens):
    # Ensure the input is properly formatted
    llm = Ollama(model=llm)
    output = StrOutputParser()
    
    # Define a system message and human message
    prompt = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=question)
    ]
    
    # Combine the chain
    chain = llm | output
    
    # Debug: Print to confirm the type of input
    print(f"Prompt Type: {type(prompt)}, Content: {prompt}")
    
    # Use the properly formatted prompt
    try:
        answer = chain.invoke(prompt)
        return answer
    except Exception as e:
        print(f"Error in chain.invoke: {e}")
        return "An error occurred while processing the request."


# In[25]:


## #Title of the app
st.title("Q&A Chatbot")


## Select the OpenAI model
llm=st.sidebar.selectbox("Select Open Source model",["mistral"])

## Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## MAin interface for user input
st.write("Go ahead and ask any question")
user_input=st.text_input("You:")



if user_input :
    response=generate_response(user_input,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide the user input")


# In[26]:





# In[ ]:




