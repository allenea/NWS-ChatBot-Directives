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
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the NWS Directives!"}
    ]
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None
if "user_region_changed" not in st.session_state:
    st.session_state.user_region_changed = False  # Track if the region has changed

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

# ✅ Region Selection UI
st.title("Welcome to the NWS Directives Chatbot")
st.write("Before we begin, please select your **NWS Region**.")

selected_region = st.selectbox(
    "Select your NWS Region:",
    [""] + list(NWS_REGIONS.values()),
    index=0 if not st.session_state.user_region else list(NWS_REGIONS.values()).index(st.session_state.user_region) + 1,
)

# ✅ Detect if the region has changed
if selected_region and selected_region != st.session_state.user_region:
    st.session_state.user_region = selected_region
    st.session_state.user_region_changed = True  # Set flag to rebuild the chat engine

if st.session_state.user_region:
    st.write(f"✅ Selected Region: **{st.session_state.user_region}**")

# ✅ Prevent chat from loading until a region is selected
if not st.session_state.user_region:
    st.warning("🚨 Please select your NWS Region to continue.")
    st.stop()

# ✅ Function to filter relevant directives
def get_filtered_documents(region):
    """Return only national directives + regional supplementals for the selected region."""
    national_docs = [doc for doc in all_docs if "National" in doc.metadata.get("region", "")]
    regional_docs = [doc for doc in all_docs if region in doc.metadata.get("region", "")]

    if not regional_docs:
        st.warning(f"⚠️ No region-specific directives found for **{region}**. Using only national directives.")

    return national_docs + regional_docs  # Ensure national directives are always included

def build_chat_engine(region, office):
    """Create a new chat engine with the correct system prompt every time the region/office changes."""
    filtered_docs = get_filtered_documents(region)  # Always filter docs correctly

    # ✅ Ensure office is always provided
    if not office:
        st.error("🚨 Office selection is required! Please select an office.")
        st.stop()

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
        llm=llm,  # Use the newly created LLM object with the correct prompt
        chat_mode="condense_question",
        verbose=True,
        streaming=True,
        return_source_nodes=True,
    )

# ✅ Force system prompt update whenever user changes region
if "chat_engine" not in st.session_state or st.session_state.user_region_changed:
    st.session_state.chat_engine = build_chat_engine(st.session_state.user_region)
    st.session_state.user_region_changed = False  # Reset change flag

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

        # ✅ Handle empty responses gracefully
        if not response_text:
            response_text = "⚠️ Sorry, I couldn't find relevant information. Try rewording your question."

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