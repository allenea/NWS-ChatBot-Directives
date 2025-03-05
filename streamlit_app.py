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
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None
if "user_selection_changed" not in st.session_state:
    st.session_state.user_selection_changed = False  # Track if region/office changes

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
    st.session_state.user_selection_changed = True

if selected_office and selected_office != st.session_state.user_office:
    st.session_state.user_office = selected_office
    st.session_state.user_selection_changed = True

if st.session_state.user_office:
    st.write(f"✅ Selected Office: **{st.session_state.user_office}**")
if st.session_state.user_region:
    st.write(f"✅ Selected Region: **{st.session_state.user_region}**")

# ✅ Prevent chat from loading until region & office are selected
if not st.session_state.user_region or not st.session_state.user_office:
    st.warning("🚨 Please select your NWS Region and Office to continue.")
    st.stop()

# ✅ Function to filter relevant directives
def get_filtered_documents(region):
    """Return only national directives + regional supplementals for the selected region."""
    national_docs = [doc for doc in all_docs if "National" in doc.metadata.get("region", "")]
    regional_docs = [doc for doc in all_docs if region in doc.metadata.get("region", "")]

    if not regional_docs:
        st.warning(f"⚠️ No region-specific directives found for **{region}**. Using only national directives.")

    return national_docs + regional_docs  # Ensure national directives are always included

# ✅ Function to build chat engine with the correct system prompt
def build_chat_engine(region, office):
    """Create a new chat engine with the correct system prompt when the region/office changes."""
    filtered_docs = get_filtered_documents(region)

    # ✅ Create a NEW OpenAI object with the correct system prompt
    llm = OpenAI(
        model="gpt-4o",
        temperature=0.2,
        system_prompt=f"""
            You are an expert on the NOAA National Weather Service (NWS) Directives. Your role is to provide
            accurate and detailed answers strictly based on official NWS and NOAA directives.

            You are assisting users from **{region}**, specifically **{office}**.
            Always tailor responses to their relevant directives.

            Guidelines for Responses:
            1. **Scope of Inquiry**:
               - Assume all questions pertain to NOAA or the National Weather Service.
            
            2. **Directive Prioritization**:
               - Prioritize **national directives** over regional supplementals unless specifically asked.
               - When citing regional supplementals, ensure they belong to the **same series and number** as the relevant national directive.

            3. **Legal Precision**:
               - Use **exact legal wording** as written in the directives (e.g., "will," "shall," "may," "should").
               - Do not interpret or modify directive language beyond what is explicitly stated.

            4. **Fact-Based Responses**:
               - Stick strictly to **documented facts**; do not hallucinate or make assumptions.
               - Always cite the **most relevant directives** when providing an answer.

            Ensure clarity, accuracy, and completeness in every response.
        """,
    )

    # ✅ Build a NEW chat engine every time with updated docs and prompt
    return VectorStoreIndex.from_documents(filtered_docs).as_chat_engine(
        llm=llm,
        chat_mode="condense_question",
        verbose=True,
        streaming=True,
        return_source_nodes=True,
    )

# ✅ Force system prompt update when region/office changes
if "chat_engine" not in st.session_state or st.session_state.user_selection_changed:
    st.session_state.chat_engine = build_chat_engine(st.session_state.user_region, st.session_state.user_office)
    st.session_state.user_selection_changed = False  # Reset flag

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
        response_text = "".join(response_stream.response_gen).strip()

        if not response_text:
            response_text = "⚠️ Sorry, I couldn't find relevant information. Try rewording your question."

        st.write(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})