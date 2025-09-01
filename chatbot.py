import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import logging
from typing import List, Dict, Any, Tuple
import PyPDF2
import re

# Vector embeddings and similarity
from sentence_transformers import SentenceTransformer
import faiss

# Groq API
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------- SESSION STATE -------------------
def init_session_state():
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'vector_index' not in st.session_state:
        st.session_state.vector_index = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

# ------------------- DOCUMENT PROCESSING -------------
class DocumentProcessor:
    """Handles document processing and text extraction"""

    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """Extract text from PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""

    @staticmethod
    def extract_text_from_txt(txt_file) -> str:
        """Extract text from TXT file"""
        try:
            return txt_file.read().decode('utf-8')
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            return ""

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap

        return chunks

# ------------------- GROQ API ------------------------
class GroqLlamaAPI:
    """Handles Groq API integration with Llama models"""

    def __init__(self, api_key: str = None):
        # Hardcoded API key
        self.api_key = "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"  # Replace this with your actual key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "moonshotai/kimi-k2-instruct"

    def generate_response(self, prompt: str, context: str = "", max_tokens: int = 1000) -> str:
        """Generate AI response"""
        if not self.api_key:
            return "Please configure Groq API key in the sidebar."

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        system_prompt = """You are a Supply Chain Document Analysis assistant.
Use the provided context to answer questions clearly and concisely."""

        full_prompt = f"{system_prompt}\n\nContext: {context}\n\nQuestion: {prompt}"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": full_prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API error: {e}")
            return f"Error calling Groq API: {e}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"Unexpected error: {e}"

# ------------------- VECTOR STORE --------------------
class VectorStore:
    """Handles embeddings and similarity search"""

    def __init__(self):
        self.model = None
        self.index = None
        self.documents = []
        self.embeddings = None

    @st.cache_resource
    def load_embedding_model(_self):
        """Load sentence transformer model"""
        return SentenceTransformer('all-MiniLM-L6-v2')

    def build_index(self, documents: List[Dict]):
        """Build vector index from documents"""
        if not documents:
            return

        self.model = self.load_embedding_model()
        self.documents = documents

        # Extract text chunks
        texts = []
        for doc in documents:
            texts.extend(doc['chunks'])

        # Generate embeddings
        with st.spinner("Generating embeddings..."):
            embeddings = self.model.encode(texts, show_progress_bar=True)

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
        self.embeddings = embeddings

        st.success(f"Vector index built with {len(texts)} text chunks")

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for relevant chunks"""
        if not self.model or not self.index:
            return []

        query_embedding = self.model.encode([query])
        scores, indices = self.index.search(query_embedding.astype('float32'), k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            chunk_count = 0
            for doc in self.documents:
                if chunk_count + len(doc['chunks']) > idx:
                    chunk_text = doc['chunks'][idx - chunk_count]
                    results.append((chunk_text, float(score)))
                    break
                chunk_count += len(doc['chunks'])

        return results

# ------------------- MAIN APP ------------------------
def main():
    st.set_page_config(
        page_title="Supply Chain Document Assistant",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Init session
    init_session_state()

    st.title("📄 Supply Chain Document Analysis Chatbot")
    st.markdown("Upload documents and chat with AI for insights.")

    # Components
    doc_processor = DocumentProcessor()
    groq_api = GroqLlamaAPI()  # Now uses the hardcoded API key
    vector_store = VectorStore()

    # Sidebar
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload Files", type=['pdf', 'txt'], accept_multiple_files=True
        )

        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                processed_docs = []
                for file in uploaded_files:
                    doc_data = {
                        'filename': file.name,
                        'type': file.type,
                        'size': len(file.getvalue()),
                        'uploaded_at': datetime.now().isoformat()
                    }

                    if file.type == 'application/pdf':
                        text = doc_processor.extract_text_from_pdf(file)
                    else:
                        text = doc_processor.extract_text_from_txt(file)

                    if text:
                        chunks = doc_processor.chunk_text(text)
                        doc_data['chunks'] = chunks
                        doc_data['text'] = text
                        processed_docs.append(doc_data)

                st.session_state.documents = processed_docs

                if processed_docs:
                    vector_store.build_index(processed_docs)
                    st.session_state.vector_index = vector_store

    # Chat interface
    if st.session_state.documents:
        st.header("💬 Chat with Your Documents")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input("Ask about your documents..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    if st.session_state.vector_index:
                        search_results = st.session_state.vector_index.search(prompt, k=3)
                        context = "\n\n".join([result[0] for result in search_results])
                    else:
                        context = ""

                    response = groq_api.generate_response(prompt, context)
                    st.write(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

    else:
        st.info("No documents yet. Upload files from sidebar to begin.")

# ------------------- RUN ----------------------------
if __name__ == "__main__":
    main()
