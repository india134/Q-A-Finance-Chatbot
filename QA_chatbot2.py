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
    ("system","You are a finance expert.Please respond to the user queries"),("user","Question:{question}")
])


# In[32]:


from langchain.memory import ChatMessageHistory
from langchain.schema import AIMessage, HumanMessage


# In[33]:


from langchain.memory import ChatMessageHistory
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Initialize chat message history (this should ideally be at a global or session level to persist across calls)
history = ChatMessageHistory()

def generate_response(question, llm, temperature, max_tokens):
    """
    Function to generate a response while maintaining chat history.
    """
    # Ensure the input is properly formatted
    output = StrOutputParser()

    # Add the user's question to history
    history.add_user_message(question)

    # Construct the prompt using chat history
    prompt = [
        SystemMessage(content="You are a helpful assistant. Always recall the user's information from the conversation history when answering questions.")
        
    ]
    # Add previous chat history to the prompt
    prompt.extend(history.messages)

    # Debug: Print the prompt and type
    print(f"Prompt Type: {type(prompt)}, Content: {prompt}")

    # Combine the chain
    chain = llm | output

    try:
        # Generate the response
        answer = chain.invoke(prompt)

        # Add the AI's response to history
        history.add_ai_message(answer)

        return answer
    except Exception as e:
        print(f"Error in chain.invoke: {e}")
        return "An error occurred while processing the request."


# In[26]:





# In[35]:


import streamlit as st
from langchain.llms import Ollama
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ChatMessageHistory


# Initialize chat history (to persist it, store it in Streamlit session state)
if "history" not in st.session_state:
    st.session_state["history"] = ChatMessageHistory()

# Title of the app
st.title("Enhanced Q&A Chatbot With OpenAI")

# Sidebar to select the model
llm = st.sidebar.selectbox("Select Open Source model", ["mistral"])

# Sidebar sliders for temperature and max tokens
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

# Define generate_response function with chat history integration
def generate_response(question, llm, temperature, max_tokens):
    """
    Generates a response while maintaining chat history.
    """
    # Ensure the input is properly formatted
    llm_model = Ollama(model=llm)
    output = StrOutputParser()

    # Add the user's question to history
    st.session_state["history"].add_user_message(question)

    # Build the prompt using chat history
    prompt = [SystemMessage(content="You are a helpful assistant.")]
    prompt.extend(st.session_state["history"].messages)

    # Combine the chain
    chain = llm_model | output

    # Generate response
    try:
        answer = chain.invoke(prompt)

        # Add AI's response to history
        st.session_state["history"].add_ai_message(answer)

        return answer
    except Exception as e:
        st.error(f"Error: {e}")
        return "An error occurred while processing your request."

# Check for user input and generate response
if user_input:
    response = generate_response(user_input, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the user input")


# In[ ]:




