# AI PDF Chatbot

A Retrieval-Augmented Generation (RAG) based PDF Chatbot built using FastAPI, Streamlit, ChromaDB, LangChain, and Groq/Ollama LLMs.

The application allows users to upload PDF documents and ask context-aware questions from the uploaded content.

---

# Features

- Upload PDF documents
- Extract and process PDF text
- Semantic search using embeddings
- Context-aware question answering
- FastAPI backend
- Streamlit frontend
- ChromaDB vector storage
- Groq / Ollama LLM support
- Research paper querying support

---

# Tech Stack

## Backend
- FastAPI
- LangChain
- ChromaDB
- PyMuPDF

## Frontend
- Streamlit

## Embeddings
- Sentence Transformers
- all-MiniLM-L6-v2

## LLM
- Groq API
- Ollama (Local LLM)

---

# Project Structure

```bash
ai-pdf-bot/
│
├── app.py
├── streamlit_app.py
├── requirements.txt
├── pdfs/
├── chroma_db/
│
├── utils/
│   ├── pdf_loader.py
│   ├── embeddings.py
│   └── retriever.py
│
└── README.md

