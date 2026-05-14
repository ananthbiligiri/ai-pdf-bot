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

```
## Sample OutPut

<img width="1918" height="917" alt="Image" src="https://github.com/user-attachments/assets/28ff7cac-8874-42f2-b2f3-af9d76abd572" />

<img width="1919" height="845" alt="Image" src="https://github.com/user-attachments/assets/d3898f9c-863f-4ef5-8349-47073e0dc6d0" />
