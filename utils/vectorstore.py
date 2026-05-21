import os

import streamlit as st

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import (
    FAISS
)

# ==================================================
# VECTOR DATABASE PATH
# ==================================================

VECTOR_DB_PATH = "vectorstore"

# ==================================================
# LOAD VECTORSTORE
# ==================================================

@st.cache_resource
def load_vectorstore(
    _embeddings
):

    # ==================================================
    # LOAD EXISTING VECTORSTORE
    # ==================================================

    if os.path.exists(
        VECTOR_DB_PATH
    ):

        return FAISS.load_local(
            VECTOR_DB_PATH,
            _embeddings,
            allow_dangerous_deserialization=True
        )

    # ==================================================
    # LOAD DOCUMENTS
    # ==================================================

    documents = []

    upload_folder = "uploads"

    pdf_files = [
        file
        for file in os.listdir(upload_folder)
        if file.endswith(".pdf")
    ]

    # ==================================================
    # READ PDF FILES
    # ==================================================

    for file in pdf_files:

        file_path = os.path.join(
            upload_folder,
            file
        )

        loader = PyPDFLoader(
            file_path
        )

        docs = loader.load()

        # ==================================================
        # ADD SOURCE METADATA
        # ==================================================

        for doc in docs:

            doc.metadata["source"] = file

        documents.extend(docs)

    # ==================================================
    # TEXT SPLITTING
    # ==================================================

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
    )

    split_docs = text_splitter.split_documents(
        documents
    )

    # ==================================================
    # CREATE VECTORSTORE
    # ==================================================

    vectorstore = FAISS.from_documents(
        split_docs,
        _embeddings
    )

    # ==================================================
    # SAVE VECTORSTORE
    # ==================================================

    vectorstore.save_local(
        VECTOR_DB_PATH
    )

    return vectorstore