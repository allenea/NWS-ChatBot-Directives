import streamlit as st
import os
import openai
import nltk
import tiktoken
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

# ✅ Initialize session state variables **before using them**
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

st.title("Welcome to the NWS Directives Chatbot")
st.write("Before we begin, please select your **NWS Office** or **Region**.")

# ✅ Dropdown to select region
selected_region = st.selectbox(
    "Select your NWS Region:",
    [""] + list(NWS_REGIONS.values()),  # Allow an empty default option
    index=0 if not st.session_state.user_region else list(NWS_REGIONS.values()).index(st.session_state.user_region) + 1,
)

# ✅ Filter offices based on selected region (or show all if none selected)
filtered_offices = [office for office, region in NWS_OFFICES.items() if region == selected_region] if selected_region else list(NWS_OFFICES.keys())

# ✅ Dropdown to select office
selected_office = st.selectbox(
    "Select your NWS Office:",
    [""] + filtered_offices,  # Allow an empty default option
    index=0 if not st.session_state.user_office else filtered_offices.index(st.session_state.user_office) + 1 if st.session_state.user_office in filtered_offices else 0,
)

# ✅ Update selections dynamically
if selected_office:
    st.session_state.user_office = selected_office
    st.session_state.user_region = NWS_OFFICES[selected_office]  # Auto-set the region based on office

if selected_region:
    st.session_state.user_region = selected_region
    st.session_state.user_office = None  # Reset office selection to prevent mismatch

# ✅ Display selected information
if st.session_state.user_office:
    st.write(f"✅ Selected Office: **{st.session_state.user_office}**")
if st.session_state.user_region:
    st.write(f"✅ Selected Region: **{st.session_state.user_region}**")

# ✅ Ensure OpenAI API key is properly loaded
if "openai_key" not in st.secrets:
    st.error("⚠️ Missing OpenAI API key! Add it to Streamlit secrets.")
    st.stop()
else:
    openai.api_key = st.secrets["openai_key"]

# ✅ Prevent chat from loading until an office or region is selected
if not st.session_state.user_office and not st.session_state.user_region:
    st.warning("🚨 Please select your NWS Office or Region to continue.")
    st.stop()

# ✅ Directives folder path
DIRECTIVES_PATH = "./directives"
if not os.path.exists(DIRECTIVES_PATH):
    st.error(f"🚨 Error: The `{DIRECTIVES_PATH}` folder is missing! Ensure it exists.")
    st.stop()

# ✅ Load and cache directive data
@st.cache_resource(show_spinner=False)
def load_data():
    """Load NWS Directives from local directory and create an index."""
    reader = SimpleDirectoryReader(input_dir=DIRECTIVES_PATH, recursive=True)
    docs = reader.load_data()

    if not docs:
        st.error("🚨 No directive documents found! Please check the 'directives' folder.")
        st.stop()

    # ✅ Use GPT-4o for high-accuracy reasoning
    Settings.llm = OpenAI(
        model="gpt-4o",
        temperature=0.2,
        system_prompt="""
            You are an expert on the NOAA National Weather Service (NWS) Directives. Your role is to provide
            accurate and detailed answers strictly based on official NWS and NOAA directives.

            Guidelines for Responses:
            1. Scope of Inquiry:
            - Assume all questions pertain to NOAA or the National Weather Service.
            
            2. Directive Prioritization:
            - Prioritize national directives over regional supplementals unless specifically asked.
            - When citing regional supplementals, ensure they belong to the same series and number as the relevant national directive.

            3. Legal Precision:
            - Use precise legal wording as written in the directives (e.g., "will," "shall," "may," "should").
            - Do not interpret or modify directive language beyond what is explicitly stated.

            4. Fact-Based Responses:
            - Stick strictly to documented facts; do not hallucinate or make assumptions.
            - Always cite the most relevant directives when providing an answer. If you aren't sure, don't cite it.

            Ensure clarity, accuracy, and completeness in every response.""",
    )

    index = VectorStoreIndex.from_documents(docs)
    return index

# ✅ Load index
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
        response_text = "".join(response_stream.response_gen)  # Collect full response before displaying
        
        # ✅ Extract relevant sources for citation
        sources = []
        max_sources = 3  # Limit to most relevant citations
        seen_sources = set()  # Avoid duplicate citations
        
        for node in response_stream.source_nodes[:max_sources]:
            source_text = node.text.strip()

            # ✅ Skip blank pages or irrelevant citations
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