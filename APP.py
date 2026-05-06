import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from langchain_groq import ChatGroq

# Load API
load_dotenv()

st.title("🤖 Advanced AI PDF Chatbot (Stable Version)")

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Upload PDFs
pdf_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if pdf_files:
    text = ""

    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()

    # Split text manually
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

    # Create embeddings
    vectors = embed_model.encode(chunks)

    # FAISS index
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors))

    st.success("PDF processed successfully ✅")

    query = st.text_input("Ask anything from PDF")

    if query:
        query_vec = embed_model.encode([query])

        _, I = index.search(np.array(query_vec), k=3)

        context = " ".join([chunks[i] for i in I[0]])

        prompt = f"Answer based on this context:\n{context}\n\nQuestion: {query}"

        response = llm.invoke(prompt)

        st.subheader("Answer:")
        st.write(response.content)