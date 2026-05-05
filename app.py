import io
import os

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder

load_dotenv()

llm = ChatGroq(groq_api_key=os.getenv('GROQ_API_KEY'),
               model = 'llama-3.1-8b-instant')

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Remember previous inputs"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)

chain = prompt | llm

#Memory store
store = {}
if "store" not in st.session_state:
    st.session_state.store = {}

store = st.session_state.store

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

#Add memory wrapper
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

st.title("Chatbot with Memory using Langchain and Groq")

if "session_id" not in st.session_state:
    st.session_state.session_id = "default"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_text" not in st.session_state:
    st.session_state["pdf_text"] = None
if "_attach_key" not in st.session_state:
    st.session_state["_attach_key"] = 0

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Composer: compact attach (no "Upload a file" label) + chat input ---
# Streamlit cannot embed upload inside st.chat_input; this mirrors ChatGPT-style attach + bar.
st.markdown(
    """
<style>
div[data-testid="stFileUploader"] {
  padding-top: 0;
  padding-bottom: 0;
}
div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
  min-height: 2.5rem !important;
  padding: 0.2rem !important;
  gap: 0 !important;
  border: none !important;
  background: transparent !important;
}
div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
  display: none !important;
}
div[data-testid="stFileUploader"] button {
  min-width: 2.3rem !important;
  width: 2.3rem !important;
  height: 2.3rem !important;
  padding: 0 !important;
  border-radius: 999px !important;
  font-size: 0 !important;
  color: transparent !important;
  text-indent: -9999px !important;
  overflow: hidden !important;
  position: relative !important;
}
div[data-testid="stFileUploader"] button * {
  font-size: 0 !important;
  color: transparent !important;
  opacity: 0 !important;
}
div[data-testid="stFileUploader"] button::after {
  content: "📎";
  font-size: 1rem;
  line-height: 1;
  color: #6b7280;
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  text-indent: 0;
}
</style>
""",
    unsafe_allow_html=True,
)


with st.form("composer_form", clear_on_submit=True):
    col_attach, col_text, col_send = st.columns([1, 9, 1], gap="small")

    with col_attach:
        uploaded_file = st.file_uploader(
            " ",
            type=["pdf", "txt"],
            label_visibility="collapsed",
            key=f"chat_attachment_{st.session_state['_attach_key']}",
        )

    with col_text:
        user_input = st.text_input(
            "",
            placeholder="Type your message here:",
            label_visibility="collapsed",
        )

    with col_send:
        submitted = st.form_submit_button("➤")

# Process attached file and store extracted text in session state
if uploaded_file is None:
    st.session_state["pdf_text"] = None
else:
    file_name = (uploaded_file.name or "").lower()
    try:
        if file_name.endswith(".pdf"):
            pdf_bytes = uploaded_file.getbuffer().tobytes()
            reader = PdfReader(io.BytesIO(pdf_bytes))
            parts = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
            full_text = "\n".join(parts)

        elif file_name.endswith(".txt"):
            raw = uploaded_file.getbuffer().tobytes()
            full_text = raw.decode("utf-8", errors="replace")

        else:
            full_text = ""
            st.warning("Use a .pdf or .txt file.")

        if full_text.strip():
            st.session_state["pdf_text"] = full_text
            with st.expander("Preview extracted text", expanded=False):
                st.text_area(
                    "First 500 characters",
                    full_text[:500],
                    height=120,
                    label_visibility="collapsed",
                )
        else:
            st.session_state["pdf_text"] = None
            st.warning("No text extracted from file.")
    except Exception as e:
        st.session_state["pdf_text"] = None
        st.error(f"Could not read file: {e}")

if submitted and user_input.strip():
    st.session_state.messages.append({'role':'user','content':user_input})
    response = chain_with_memory.invoke({'input':user_input},
                                        config={'configurable':{'session_id':st.session_state.session_id}})
    bot_reply = response.content
    st.session_state.messages.append({'role':'assistant','content':bot_reply})
    with st.chat_message("assistant"):
        st.write(bot_reply)
    

if st.button("Clear Conversation"):
    st.session_state.messages = []
    st.session_state["pdf_text"] = None
    st.session_state["_attach_key"] += 1
    store.clear()
