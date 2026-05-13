from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

import os
import shutil
import uuid

# =========================
# LOAD ENV
# =========================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
CHROMA_DB_BASE_DIR = "chroma_db"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DB_BASE_DIR, exist_ok=True)

# =========================
# GLOBAL STATE
# =========================

current_db_path = None

# =========================
# EMBEDDING MODEL
# =========================

# Embedding model moved inside endpoints for faster startup

# =========================
# LOAD VECTOR DB
# =========================

# Vector DB moved inside endpoints for faster startup

# =========================
# GEMINI LLM
# =========================



llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
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

    global current_db_path

    try:

        # Initialize embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

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

        # Generate unique DB path for this upload
        current_db_path = os.path.join(CHROMA_DB_BASE_DIR, str(uuid.uuid4()))

        # Create vector DB
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=current_db_path
        )

        # Clean up
        del embedding_model
        del vector_db

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

    try:

        if current_db_path is None:

            return JSONResponse(
                status_code=400,
                content={
                    "error": "Upload PDF first"
                }
            )

        # Initialize embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load vector DB
        vector_db = Chroma(
            persist_directory=current_db_path,
            embedding_function=embedding_model
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
        response = qa_chain.invoke({"query": question})
        answer = response["result"]

        # Clean up
        del embedding_model
        del vector_db

        return JSONResponse(
            content={
                "question": question,
                "answer": answer
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

    global current_db_path

    try:

        # Remove all ChromaDB directories
        if os.path.exists(CHROMA_DB_BASE_DIR):

            shutil.rmtree(CHROMA_DB_BASE_DIR)

        os.makedirs(CHROMA_DB_BASE_DIR, exist_ok=True)

        current_db_path = None

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

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(app, host="0.0.0.0", port=port)