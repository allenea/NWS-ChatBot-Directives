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

st.title("Welcome to the NWS Directives Chatbot")
st.write("Before we begin, please select your **NWS Office** or **Region**.")

# ✅ Dropdown to select region
selected_region = st.selectbox(
    "Select your NWS Region:",
    [""] + list(NWS_REGIONS.values()),  
    index=0 if not st.session_state.user_region else list(NWS_REGIONS.values()).index(st.session_state.user_region) + 1,
)

# ✅ Filter offices based on selected region
filtered_offices = [office for office, region in NWS_OFFICES.items() if region == selected_region] if selected_region else list(NWS_OFFICES.keys())

# ✅ Dropdown to select office
selected_office = st.selectbox(
    "Select your NWS Office:",
    [""] + filtered_offices,  
    index=0 if not st.session_state.user_office else filtered_offices.index(st.session_state.user_office) + 1 if st.session_state.user_office in filtered_offices else 0,
)

# ✅ Update session state dynamically
if selected_office:
    st.session_state.user_office = selected_office
    st.session_state.user_region = NWS_OFFICES[selected_office]

if selected_region:
    st.session_state.user_region = selected_region
    st.session_state.user_office = None  

# ✅ Display selection
if st.session_state.user_office:
    st.write(f"✅ Selected Office: **{st.session_state.user_office}**")
if st.session_state.user_region:
    st.write(f"✅ Selected Region: **{st.session_state.user_region}**")

# ✅ Ensure OpenAI API key exists
if "openai_key" not in st.secrets:
    st.error("⚠️ Missing OpenAI API key! Add it to Streamlit secrets.")
    st.stop()
else:
    openai.api_key = st.secrets["openai_key"]

# ✅ Prevent chat from loading until an office or region is selected
if not st.session_state.user_office and not st.session_state.user_region:
    st.warning("🚨 Please select your NWS Office or Region to continue.")
    st.stop()

# ✅ Define directives path
DIRECTIVES_PATH = "./directives"
if not os.path.exists(DIRECTIVES_PATH):
    st.error(f"🚨 Error: The `{DIRECTIVES_PATH}` folder is missing!")
    st.stop()

# ✅ Load and cache directive data
@st.cache_resource(show_spinner=False)
def load_directives(region):
    """Load directives while filtering for national and relevant regional directives."""
    reader = SimpleDirectoryReader(input_dir=DIRECTIVES_PATH, recursive=True)
    all_docs = reader.load_data()

    if not all_docs:
        st.error("🚨 No directive documents found!")
        st.stop()

    # ✅ Filter directives
    national_directives = [doc for doc in all_docs if "pd" in doc.metadata["file_name"]]
    regional_directives = [doc for doc in all_docs if region.lower() in doc.metadata["file_name"].lower()]

    # ✅ Debugging Information
    total_docs = len(all_docs)
    national_count = len(national_directives)
    regional_count = len(regional_directives)

    st.write(f"📊 **Debugging Info:**")
    st.write(f"- Total directives loaded: **{total_docs}**")
    st.write(f"- National directives: **{national_count}**")
    st.write(f"- Regional supplementals for `{region}`: **{regional_count}**")

    # ✅ Combine documents for indexing
    filtered_docs = national_directives + regional_directives

    # ✅ Use GPT-4o for high-accuracy reasoning
    Settings.llm = OpenAI(
        model="gpt-4o",
        temperature=0.2,
        system_prompt=f"""
            You are an expert on the NOAA National Weather Service (NWS) Directives. Your role is to provide
            accurate and detailed answers based strictly on official NWS and NOAA directives.

            You understand the classification rules for regional supplementals as defined in the document 
            'pd00101001curr.pdf'. Use these rules to determine which regional directives apply to {st.session_state.user_region}, in 
            addition to always considering national directives.

            Guidelines:
            1. Assume all questions relate to NOAA or the National Weather Service.
            2. Prioritize national directives over regional supplementals unless specifically asked.
            3. When citing regional supplementals, ensure the national directive for that series and directive number is also included.
            4. Use precise legal wording as written in the directives (e.g., "will," "shall," "may," "should").
            5. Do not interpret or modify directive language beyond what is explicitly stated.
            6. Always cite the most relevant directive in responses.
            7. Stick strictly to documented facts; do not make assumptions. Do not hallucinate.""",
    )

    index = VectorStoreIndex.from_documents(filtered_docs)
    return index

# ✅ Load directives based on user selection
index = load_directives(st.session_state.user_region)

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
        response_text = "".join(response_stream.response_gen) 

        # ✅ Extract sources
        sources = []
        seen_sources = set()
        for node in response_stream.source_nodes[:3]:  
            if "file_name" in node.metadata:
                file_name = os.path.basename(node.metadata["file_name"])
                source_url = f"https://www.weather.gov/media/directives/{file_name}"
                if source_url not in seen_sources:
                    sources.append(f"- {file_name} ([View]({source_url}))")
                    seen_sources.add(source_url)

        if sources:
            response_text += "\n\n**Sources:**\n" + "\n".join(sources)

        st.write(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})