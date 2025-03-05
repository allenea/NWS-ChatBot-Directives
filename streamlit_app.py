import streamlit as st
import os
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from nws_options import NWS_OFFICES, NWS_REGIONS  # Import NWS options

# ✅ Set Streamlit page configuration
st.set_page_config(
    page_title="Chat with the NWS Directives, powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto"
)

# ✅ Initialize session state variables
if "user_region" not in st.session_state:
    st.session_state.user_region = None
if "user_office" not in st.session_state:
    st.session_state.user_office = None
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the NWS Directives!"}
    ]
if "index" not in st.session_state:
    st.session_state.index = None  # Store full index of directives
if "classification_doc" not in st.session_state:
    st.session_state.classification_doc = None  # Stores the supplemental classification document
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None  # Stores the LLM chat engine

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
    """Load all NWS Directives and store classification rules separately."""
    reader = SimpleDirectoryReader(input_dir=DIRECTIVES_PATH, recursive=True)
    
    all_docs = reader.load_data()

    # ✅ Ensure classification rules document is loaded first
    classification_doc = next((doc for doc in all_docs if "pd00101001curr.pdf" in doc.metadata.get("file_path", "")), None)

    if classification_doc:
        st.write("✅ Classification rules for regional supplementals loaded successfully!")
    else:
        st.error("🚨 Could not find 'pd00101001curr.pdf'. Ensure it exists in the directives folder!")

    if not all_docs:
        st.error("🚨 No directive documents found! Please check the 'directives' folder.")
        st.stop()

    # ✅ Build the index once and store it persistently
    index = VectorStoreIndex.from_documents(all_docs)

    return index, classification_doc

# ✅ Load directives once and store them
if "index" not in st.session_state or "classification_doc" not in st.session_state:
    st.session_state.index, st.session_state.classification_doc = load_all_directives()

# ✅ Region & Office Selection UI
st.title("Welcome to the NWS Directives Chatbot")
st.write("Before we begin, please select your **NWS Region** and **Office**.")

# Region selection
selected_region = st.selectbox(
    "Select your NWS Region:",
    [""] + list(NWS_REGIONS.values()),
    index=0 if not st.session_state.user_region else list(NWS_REGIONS.values()).index(st.session_state.user_region) + 1,
)

# Filter offices based on region selection
filtered_offices = [office for office, region in NWS_OFFICES.items() if region == selected_region] if selected_region else list(NWS_OFFICES.keys())

# Office selection
selected_office = st.selectbox(
    "Select your NWS Office:",
    [""] + filtered_offices,
    index=0 if not st.session_state.user_office else filtered_offices.index(st.session_state.user_office) + 1 if st.session_state.user_office in filtered_offices else 0,
)

# ✅ Detect if the selection has changed
if selected_region and selected_region != st.session_state.user_region:
    st.session_state.user_region = selected_region

if selected_office and selected_office != st.session_state.user_office:
    st.session_state.user_office = selected_office

if st.session_state.user_office:
    st.write(f"✅ Selected Office: **{st.session_state.user_office}**")
if st.session_state.user_region:
    st.write(f"✅ Selected Region: **{st.session_state.user_region}**")

# ✅ Prevent chat from loading until region & office are selected
if not st.session_state.user_region or not st.session_state.user_office:
    st.warning("🚨 Please select your NWS Region and Office to continue.")
    st.stop()

# ✅ Function to retrieve relevant documents at query time
def get_relevant_documents(query, region):
    """Retrieve only relevant directives for the given query and region."""
    query_engine = st.session_state.index.as_query_engine()

    # ✅ Retrieve documents based on user query
    retrieved_docs = query_engine.query(query)

    # ✅ Filter results to include only the selected region's directives
    filtered_docs = [doc for doc in retrieved_docs if doc.metadata.get("region", "") == "National" or doc.metadata.get("region", "") == region]

    st.write(f"🔍 Retrieved {len(filtered_docs)} relevant documents for {region}")

    return filtered_docs

# ✅ Function to create the chat engine with a proper system prompt
def build_chat_engine():
    """Create the chat engine once at startup and reuse it."""
    system_prompt = f"""
        You are an expert on the NOAA National Weather Service (NWS) Directives. Your role is to provide
        accurate and detailed answers based strictly on official NWS and NOAA directives.

        You understand the classification rules for regional supplementals as defined in the document 
        'pd00101001curr.pdf'. Use these rules to determine which regional directives apply to {st.session_state.user_region}, in 
        addition to always considering national directives.

        Guidelines:
        1. Assume all questions relate to NOAA or the National Weather Service.
        2. Provide expert interpretation and reasoning.
        3. Prioritize national directives, add additional context from associated regional supplementals, where appropriate,
           unless specifically asked about a specific national directive or regional supplemental.
        3. When citing regional supplementals, ensure the national directive for that series and directive number is also.
        4. Use precise legal wording as written in the directives (e.g., "will," "shall," "may," "should").
        5. Do not interpret or modify directive language beyond what is explicitly stated.
        6. Always cite the most relevant directive in responses.
        7. Stick strictly to documented facts; do not make assumptions. Do not hallucinate.
    """

    llm = OpenAI(
        model="gpt-4o",
        temperature=0.2,
        system_prompt=system_prompt,
    )

    return llm

# ✅ Initialize chat engine once
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = build_chat_engine()

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
        query = st.session_state.messages[-1]["content"]
        relevant_docs = get_relevant_documents(query, st.session_state.user_region)
        response = st.session_state.chat_engine.chat(query)
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})