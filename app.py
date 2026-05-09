import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

# =========================
# LOAD ENV
# =========================

load_dotenv()

groq_api_key = os.getenv(
    "GROQ_API_KEY"
)

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Healthcare RAG Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title(
    "🩺 Healthcare RAG Assistant"
)

st.info(
    """
Example Questions:

• What are symptoms of dengue?
• How is diabetes diagnosed?
• Compare malaria and dengue symptoms.
• What treatment is recommended for hypertension?
"""
)

# =========================
# EMBEDDINGS
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# VECTOR DB CREATION
# =========================

VECTOR_DB_PATH = "vectorstore"

@st.cache_resource
def load_vectorstore():

    # If vector DB already exists
    if os.path.exists(VECTOR_DB_PATH):

        vectorstore = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vectorstore

    # Else create vector DB

    documents = []

    upload_folder = "uploads"

    pdf_files = [
        file
        for file in os.listdir(upload_folder)
        if file.endswith(".pdf")
    ]

    for file in pdf_files:

        file_path = os.path.join(
            upload_folder,
            file
        )

        loader = PyPDFLoader(
            file_path
        )

        docs = loader.load()

        for doc in docs:

            doc.metadata["source"] = file

        documents.extend(docs)

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    )

    split_docs = text_splitter.split_documents(
        documents
    )

    vectorstore = FAISS.from_documents(
        split_docs,
        embeddings
    )

    vectorstore.save_local(
        VECTOR_DB_PATH
    )

    return vectorstore

# =========================
# LOAD VECTORSTORE
# =========================

with st.spinner(
    "Loading Healthcare Knowledge Base..."
):

    vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# =========================
# LLM
# =========================

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# =========================
# QUESTION INPUT
# =========================

question = st.text_input(
    "Ask a healthcare question:"
)

# =========================
# ANSWERING
# =========================

if question:

    with st.spinner(
        "Generating answer..."
    ):

        retrieved_docs = retriever.invoke(
            question
        )

        context = "\n\n".join(
            [
                doc.page_content
                for doc in retrieved_docs
            ]
        )

        sources = list(
            set(
                [
                    f"{doc.metadata.get('source')} (Page {doc.metadata.get('page')})"
                    for doc in retrieved_docs
                ]
            )
        )

        prompt = f"""
You are a healthcare AI assistant.

Answer ONLY from the given context.

If information is unavailable,
say:

'The information is not available in the uploaded healthcare documents.'

Context:
{context}

Question:
{question}
"""

        response = llm.invoke(
            prompt
        )

        st.subheader("Answer")

        st.write(
            response.content
        )

        st.subheader("Sources")

        for source in sources:

            st.write(source)