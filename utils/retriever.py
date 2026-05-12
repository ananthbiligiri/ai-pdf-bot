from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import os

# Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def get_qa_chain(vector_db):

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
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

    return qa_chain