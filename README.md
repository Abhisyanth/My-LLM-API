# GenAI Chatbot with Memory

A conversational AI chatbot built with LangChain and Groq that maintains session-based memory — allowing it to remember previous messages and respond contextually throughout a conversation.

## Live Demo
https://my-llm-api-foecy.streamlit.app/

## Project Overview
Most basic chatbots treat every message independently — they forget what was said earlier in the conversation. This chatbot solves that by maintaining conversation history using LangChain's memory management, making interactions feel natural and context-aware.

## Features
- Interactive chat UI built with Streamlit
- Conversation memory using LangChain RunnableWithMessageHistory
- Fast responses powered by Groq LLM (Llama 3.1)
- Chat history displayed in the UI
- Clear conversation button to reset session

## Tech Stack
- Python
- Streamlit
- LangChain
- Groq (Llama 3.1)

## Installation and Setup

```bash
git clone https://github.com/Abhisyanth/My-LLM-API
cd My-LLM-API
pip install -r requirements.txt
```

## How It Works
1. User types a message in the chat input
2. LangChain passes the message along with conversation history to Groq LLM
3. Groq processes the full context and generates a response
4. Response is displayed in the chat UI
5. Conversation history is updated for the next message
6. User can clear the conversation at any time to start fresh

## Future Improvements
- Add support for multiple LLM providers (OpenAI, Anthropic)
- Allow users to upload documents and chat with them
- Persistent conversation history across sessions using a database
- Add system prompt customisation for different use cases
- Export conversation as text or PDF

## GitHub
https://github.com/Abhisyanth/My-LLM-API

**Running locally:**

Create a `.env` file in the root folder and add:
