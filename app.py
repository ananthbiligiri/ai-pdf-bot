from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyMuPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

import os
import shutil

# =========================
# LOAD ENV
# =========================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# =========================
# FASTAPI APP
# =========================

app = FastAPI(
    title="PDF Chatbot API",
    version="1.0"
)

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CONFIG
# =========================

PDF_FOLDER = "pdfs"
CHROMA_DB_DIR = "chroma_db"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# =========================
# EMBEDDING MODEL
# =========================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# LOAD VECTOR DB
# =========================

vector_db = None

if os.listdir(CHROMA_DB_DIR):

    vector_db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding_model
    )

# =========================
# GEMINI LLM
# =========================

llm = ChatOllama(
    model="gemma2:2b",
    temperature=0
)

# =========================
# ROOT
# =========================

@app.get("/")
async def root():

    return {
        "message": "PDF Chatbot API Running"
    }

# =========================
# HEALTH CHECK
# =========================

@app.get("/health")

async def health_check():

    return {
        "status": "healthy"
    }

# =========================
# UPLOAD PDF
# =========================

@app.post("/upload-pdf")

async def upload_pdf(file: UploadFile = File(...)):

    global vector_db

    try:

        # Validate PDF
        if not file.filename.endswith(".pdf"):

            return JSONResponse(
                status_code=400,
                content={
                    "error": "Only PDF files are allowed"
                }
            )

        # Save PDF
        file_path = os.path.join(
            PDF_FOLDER,
            file.filename
        )

        with open(file_path, "wb") as buffer:

            shutil.copyfileobj(file.file, buffer)

        # Load PDF
        loader = PyMuPDFLoader(file_path)

        documents = loader.load()

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        # Create vector DB
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=CHROMA_DB_DIR
        )

        vector_db.persist()

        return JSONResponse(
            content={
                "message": f"{file.filename} uploaded successfully",
                "chunks": len(chunks)
            }
        )

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )

# =========================
# ASK QUESTION
# =========================

@app.post("/ask")

async def ask_question(question: str = Form(...)):

    global vector_db

    try:

        # Check DB
        if vector_db is None:

            return JSONResponse(
                status_code=400,
                content={
                    "error": "Upload PDF first"
                }
            )

        # Retriever
        retriever = vector_db.as_retriever(
            search_kwargs={"k": 3}
        )

        # QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )

        # Generate answer
        response = qa_chain.run(question)

        return JSONResponse(
            content={
                "question": question,
                "answer": response
            }
        )

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )

# =========================
# CLEAR DATABASE
# =========================

@app.delete("/clear-db")

async def clear_database():

    global vector_db

    try:

        # Remove ChromaDB files
        if os.path.exists(CHROMA_DB_DIR):

            shutil.rmtree(CHROMA_DB_DIR)

        os.makedirs(CHROMA_DB_DIR)

        vector_db = None

        return {
            "message": "Database cleared successfully"
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )