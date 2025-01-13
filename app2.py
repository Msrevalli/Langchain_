# Import necessary libraries
from langchain_openai import ChatOpenAI  # OpenAI chat model integration
from langchain_core.prompts import ChatPromptTemplate  # For structuring chat prompts
from langchain_core.output_parsers import StrOutputParser  # Converts output to string
import streamlit as st  # Web app framework
import os  # For environment variables
from dotenv import load_dotenv  # For loading environment variables from .env file

# Load environment variables from .env file
load_dotenv()

# Set up environment variables for API keys and tracing
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Enable LangChain tracing

# Update the prompt template to include chat history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions about the user's input. Maintain context from the chat history provided."),
    ("system", "Previous conversation:\n{chat_history}"),
    ("user", "{input}")
])

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # temperature=0 for more deterministic responses

# Create a chain: prompt -> LLM -> String output
chain = prompt | llm | StrOutputParser()

# Set up Streamlit interface
st.title("ChatGPT Replica by Sreevalli")
st.write("Ask me anything!")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages in chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle new user input
if prompt := st.chat_input("Enter your message"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Format chat history for the prompt
    chat_history = "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in st.session_state.messages[:-1]  # Exclude the current message
    ])
    
    # Get AI response with chat history context
    result = chain.invoke({
        "input": prompt,
        "chat_history": chat_history if chat_history else "No previous messages"
    })
    
    # Add AI response to history
    st.session_state.messages.append({"role": "assistant", "content": result})
    
    # Display AI response
    with st.chat_message("assistant"):
        st.write(result)