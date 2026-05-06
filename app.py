import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Remember previous inputs."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

chain = prompt | llm

if "store" not in st.session_state:
    st.session_state.store = {}
store = st.session_state.store


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

st.title("Chatbot with Memory using Langchain and Groq")

if "session_id" not in st.session_state:
    st.session_state.session_id = "default"

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Type your message here:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    response = chain_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": st.session_state.session_id}},
    )
    bot_reply = response.content

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.write(bot_reply)

if st.button("Clear Conversation"):
    st.session_state.messages = []
    store.clear()
