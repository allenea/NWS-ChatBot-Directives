import os
import streamlit as st
import openai
import nltk
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

# Fix: Set a writable directory for NLTK data
os.environ['NLTK_DATA'] = '/tmp/nltk_data'
nltk.data.path.append('/tmp/nltk_data')
nltk.download("stopwords", download_dir='/tmp/nltk_data', quiet=True)

# Streamlit page settings
st.set_page_config(
    page_title="Chat with the NWS Directives, powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("Chat with the NWS Directives")

# Ensure OpenAI API key is properly loaded
if "openai_key" not in st.secrets:
    st.error("Missing OpenAI API key in secrets!")
else:
    openai.api_key = st.secrets["openai_key"]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the NWS Directives!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    """Load NWS Directives from local directory and create an index."""
    reader = SimpleDirectoryReader(input_dir="./directives", recursive=True)
    docs = reader.load_data()
    
    # Set up LlamaIndex with OpenAI
    Settings.llm = OpenAI(
        model="gpt-4o",
        temperature=0.2,
        system_prompt="""You are an expert on the NOAA National Weather Service Directives.
        Your job is to answer detailed questions based on official documents.
        - Assume all questions relate to NOAA or the National Weather Service.
        - Prioritize national directives over regional supplementals unless specifically asked.
        - Be precise with legal wording (e.g., will, shall, may, should).
        - Stick to facts; do not hallucinate or make assumptions.
        """,
    )
    
    index = VectorStoreIndex.from_documents(docs)
    return index

# Load index
index = load_data()

# Initialize chat engine
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

# Get user input
if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Generate response if last message is from user
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        st.session_state.messages.append(
            {"role": "assistant", "content": response_stream.response}
        )