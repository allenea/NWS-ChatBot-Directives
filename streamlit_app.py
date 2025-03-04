import streamlit as st
import os
import openai
import nltk
import tiktoken
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

# ✅ `st.set_page_config()` MUST be the first Streamlit command
st.set_page_config(
    page_title="Chat with the NWS Directives, powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto"
)

# ✅ Initialize session state variables **before using them**
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the NWS Directives!"}
    ]

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None

st.write("🚀 App is starting...")  # Debugging message

# ✅ Fix: Set writable directories for caching
TOKEN_CACHE_PATH = "/tmp/tiktoken_cache"
LLAMA_CACHE_PATH = "/tmp/llama_index_cache"
NLTK_DATA_PATH = "/tmp/nltk_data"

os.environ["TIKTOKEN_CACHE_DIR"] = TOKEN_CACHE_PATH
os.environ["LLAMA_INDEX_CACHE_DIR"] = LLAMA_CACHE_PATH
os.environ["NLTK_DATA"] = NLTK_DATA_PATH

os.makedirs(TOKEN_CACHE_PATH, exist_ok=True)
os.makedirs(LLAMA_CACHE_PATH, exist_ok=True)
os.makedirs(NLTK_DATA_PATH, exist_ok=True)

# ✅ Prevent Tiktoken from caching in restricted directories
tiktoken.TIKTOKEN_CACHE_DIR = TOKEN_CACHE_PATH

# ✅ Fix: Ensure NLTK stopwords are available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=NLTK_DATA_PATH, quiet=True)

st.title("Chat with the NWS Directives")

# ✅ Ensure OpenAI API key is properly loaded
if "openai_key" not in st.secrets:
    st.error("⚠️ Missing OpenAI API key! Add it to Streamlit secrets.")
    st.stop()
else:
    openai.api_key = st.secrets["openai_key"]

# ✅ Check if Directives Folder Exists
DIRECTIVES_PATH = "./directives"
if not os.path.exists(DIRECTIVES_PATH):
    st.error(f"🚨 Error: The `{DIRECTIVES_PATH}` folder is missing! Ensure it exists.")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_data():
    """Load NWS Directives from local directory and create an index."""
    reader = SimpleDirectoryReader(input_dir=DIRECTIVES_PATH, recursive=True)
    docs = reader.load_data()

    if not docs:
        st.error("🚨 No directive documents found! Please check the 'directives' folder.")
        st.stop()

    # ✅ Use GPT-4o for high accuracy reasoning
    Settings.llm = OpenAI(
        model="gpt-4o",
        temperature=0.2,
        system_prompt="""
        You are an expert on the NOAA National Weather Service Directives.
        Your job is to answer detailed questions based on official documents.
        - Assume all questions relate to NOAA or the National Weather Service.
        - Prioritize national directives over regional supplementals unless specifically asked.
        - Be precise with legal wording (e.g., will, shall, may, should).
        - Always cite the most relevant directive in your answer.
        - Stick to facts; do not hallucinate or make assumptions.
        """,
    )

    index = VectorStoreIndex.from_documents(docs)
    return index

# ✅ Load index (only if data is available)
index = load_data()

# ✅ Initialize chat engine
if st.session_state.chat_engine is None:
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True, return_source_nodes=True
    )

# ✅ Get user input
if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# ✅ Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ✅ Generate response if last message is from user
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        
        # ✅ Extract sources for citation and format as hyperlinks
        sources = []
        max_sources = 5  # Limit the number of citations
        for node in response_stream.source_nodes[:max_sources]:
            source_text = node.text[:200]  # Show first 200 characters of the source
            
            # Check if metadata contains a valid source reference
            source_url = "Unknown Source"
            if "file_name" in node.metadata:
                file_name = os.path.basename(node.metadata["file_name"])
                
                # Extract the directive series (e.g., "020" from "pd02001003e042003curr.pdf")
                series_match = file_name[2:5] if file_name.startswith("pd") and file_name[2:5].isdigit() else None
                if series_match:
                    source_url = f"https://www.weather.gov/media/directives/{series_match}_pdfs/{file_name}"
            
            sources.append(f"- [{source_text}...]({source_url})")  # Hyperlink source

        # ✅ Capture and display the response immediately
        response_text = response_stream.response

        # ✅ Append sources at the end (instead of replacing response)
        if sources:
            response_text += "\n\n**Sources:**\n" + "\n".join(sources)
        
        st.write(response_text)  # Display response with citations immediately
        
        # ✅ Add response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text})
