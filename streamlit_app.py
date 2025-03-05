import streamlit as st
import os
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from nws_options import NWS_OFFICES, NWS_REGIONS  # Import from your NWS options file

# ✅ Set Streamlit page configuration
st.set_page_config(
    page_title="Chat with the NWS Directives, powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto"
)

# ✅ Initialize session state variables
if "user_office" not in st.session_state:
    st.session_state.user_office = None
if "user_region" not in st.session_state:
    st.session_state.user_region = None
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the NWS Directives!"}
    ]
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None

# ✅ Ensure OpenAI API key is properly loaded
if "openai_key" not in st.secrets:
    st.error("⚠️ Missing OpenAI API key! Add it to Streamlit secrets.")
    st.stop()
else:
    openai.api_key = st.secrets["openai_key"]

# ✅ Load all directives at startup
DIRECTIVES_PATH = "./directives"
if not os.path.exists(DIRECTIVES_PATH):
    st.error(f"🚨 Error: The `{DIRECTIVES_PATH}` folder is missing! Ensure it exists.")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_all_directives():
    """Load all NWS Directives (National + Regional) at app startup."""
    reader = SimpleDirectoryReader(input_dir=DIRECTIVES_PATH, recursive=True)
    docs = reader.load_data()

    if not docs:
        st.error("🚨 No directive documents found! Please check the 'directives' folder.")
        st.stop()

    return docs  # Store all docs in memory for filtering later

# ✅ Load full dataset once
all_docs = load_all_directives()

# ✅ Office/Region Selection UI
st.title("Welcome to the NWS Directives Chatbot")
st.write("Before we begin, please select your **NWS Office** or **Region**.")

selected_region = st.selectbox(
    "Select your NWS Region:",
    [""] + list(NWS_REGIONS.values()),
    index=0 if not st.session_state.user_region else list(NWS_REGIONS.values()).index(st.session_state.user_region) + 1,
)

filtered_offices = [office for office, region in NWS_OFFICES.items() if region == selected_region] if selected_region else list(NWS_OFFICES.keys())

selected_office = st.selectbox(
    "Select your NWS Office:",
    [""] + filtered_offices,
    index=0 if not st.session_state.user_office else filtered_offices.index(st.session_state.user_office) + 1 if st.session_state.user_office in filtered_offices else 0,
)

if selected_office:
    st.session_state.user_office = selected_office
    st.session_state.user_region = NWS_OFFICES[selected_office]

if selected_region:
    st.session_state.user_region = selected_region
    st.session_state.user_office = None  # Reset office to prevent mismatches

if st.session_state.user_office:
    st.write(f"✅ Selected Office: **{st.session_state.user_office}**")
if st.session_state.user_region:
    st.write(f"✅ Selected Region: **{st.session_state.user_region}**")

# ✅ Prevent chat from loading until a region is selected
if not st.session_state.user_region:
    st.warning("🚨 Please select your NWS Office or Region to continue.")
    st.stop()

# ✅ Filter directives dynamically based on selected region
@st.cache_resource(show_spinner=False)
def get_filtered_index(region):
    """Filter out only relevant regional supplementals while keeping national directives."""
    national_docs = [doc for doc in all_docs if "National" in doc.metadata.get("region", "")]
    regional_docs = [doc for doc in all_docs if region in doc.metadata.get("region", "")]

    combined_docs = national_docs + regional_docs  # Keep national directives + region-specific supplementals

    if not regional_docs:
        st.warning(f"⚠️ No region-specific directives found for **{region}**. Only national directives will be used.")

    return VectorStoreIndex.from_documents(combined_docs)

# ✅ Load the filtered index based on selected region
filtered_index = get_filtered_index(st.session_state.user_region)

# ✅ Initialize chat engine only if not set
if st.session_state.chat_engine is None:
    st.session_state.chat_engine = filtered_index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True, return_source_nodes=True
    )

st.write("---")
st.title("Chat with the NWS Directives")
st.write("Ask me a question about the NWS Directives!")

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
        response_text = "".join(response_stream.response_gen)  # Collect full response before displaying
        
        # ✅ Extract relevant sources for citation
        sources = []
        max_sources = 3  # Limit to most relevant citations
        seen_sources = set()
        
        for node in response_stream.source_nodes[:max_sources]:
            source_text = node.text.strip()
            if not source_text or "This page intentionally left blank" in source_text:
                continue

            source_excerpt = source_text[:200].strip()
            if len(source_excerpt) < 20:
                continue

            source_url = "Unknown Source"
            if "file_name" in node.metadata:
                file_name = os.path.basename(node.metadata["file_name"])
                series_match = file_name[2:5] if file_name.startswith("pd") and file_name[2:5].isdigit() else None
                if series_match:
                    source_url = f"https://www.weather.gov/media/directives/{series_match}_pdfs/{file_name}"

            if source_url not in seen_sources:
                sources.append(f"- [{source_excerpt}...]({source_url})")
                seen_sources.add(source_url)

        if sources:
            response_text += "\n\n**Sources:**\n" + "\n".join(sources)

        st.write(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})