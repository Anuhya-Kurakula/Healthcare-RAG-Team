import os
import shutil
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
    layout="wide"
)

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.header(
        "Healthcare RAG Assistant"
    )

    st.write(
        "Ask questions from healthcare PDFs."
    )

    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.rerun()

# =========================
# TITLE
# =========================

st.title(
    "Healthcare RAG Assistant"
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
# VECTOR DB PATH
# =========================

VECTOR_DB_PATH = "vectorstore"

# =========================
# PDF UPLOAD
# =========================

uploaded_files = st.file_uploader(
    "Upload Healthcare PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    # Create uploads folder

    os.makedirs(
        "uploads",
        exist_ok=True
    )

    # Delete old uploaded PDFs

    for old_file in os.listdir("uploads"):

        old_path = os.path.join(
            "uploads",
            old_file
        )

        os.remove(old_path)

    # Save new uploaded PDFs

    for uploaded_file in uploaded_files:

        save_path = os.path.join(
            "uploads",
            uploaded_file.name
        )

        with open(save_path, "wb") as f:

            f.write(uploaded_file.read())

    # Delete old vector database

    if os.path.exists(VECTOR_DB_PATH):

        shutil.rmtree(VECTOR_DB_PATH)

    # Clear cache

    st.cache_resource.clear()

    st.success(
        "PDFs uploaded successfully!"
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

    if not os.path.exists(upload_folder):

        os.makedirs(upload_folder)

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

    # Text Splitting

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    )

    split_docs = text_splitter.split_documents(
        documents
    )

    # Create FAISS DB

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
# SESSION STATE
# =========================

if "messages" not in st.session_state:

    st.session_state.messages = []

# =========================
# DISPLAY CHAT HISTORY
# =========================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

# =========================
# CHAT INPUT
# =========================

question = st.chat_input(
    "Ask a healthcare question"
)

# =========================
# ANSWERING
# =========================

if question:

    # Store User Message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Display User Message

    with st.chat_message("user"):

        st.markdown(question)

    with st.spinner(
        "Generating answer..."
    ):

        # Retrieve Docs

        retrieved_docs = retriever.invoke(
            question
        )

        # Context Creation

        context = "\n\n".join(
            [
                doc.page_content
                .replace("z", "•")
                .replace("\n", " ")
                for doc in retrieved_docs
            ]
        )

        # Sources

        sources = list(
            set(
                [
                    f"{doc.metadata.get('source')} (Page {doc.metadata.get('page')})"
                    for doc in retrieved_docs
                ]
            )
        )

        # Prompt

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

        # LLM Response

        response = llm.invoke(
            prompt
        )

        # Assistant Response

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                response.content
            )

            st.subheader(
                "Sources"
            )

            for source in sources:

                st.write(source)

    # Store Assistant Response

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.content
        }
    )