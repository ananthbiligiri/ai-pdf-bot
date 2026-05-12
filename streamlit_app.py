import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI PDF Chatbot")
st.write("Upload a PDF and ask questions")

# =========================
# SESSION STATE
# =========================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================
# PDF UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    with st.spinner("Uploading and processing PDF..."):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file,
                "application/pdf"
            )
        }

        response = requests.post(
            f"{API_URL}/upload-pdf",
            files=files
        )

        if response.status_code == 200:
            result = response.json()

            st.success(result["message"])
            st.info(f"Chunks Created: {result['chunks']}")

        else:
            st.error("Failed to upload PDF")

# =========================
# QUESTION INPUT
# =========================

question = st.chat_input("Ask a question from PDF")

if question:

    # Save user message
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.spinner("Thinking..."):

        response = requests.post(
            f"{API_URL}/ask",
            data={"question": question}
        )

        if response.status_code == 200:

            answer = response.json()["answer"]

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        else:
            st.error("Error generating answer")

# =========================
# DISPLAY CHAT
# =========================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])