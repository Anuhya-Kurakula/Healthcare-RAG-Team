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

from langchain.memory import (
    ConversationBufferMemory
)

from langchain.chains import (
    ConversationalRetrievalChain
)

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

    os.makedirs(
        "uploads",
        exist_ok=True
    )

    # Check if new upload exists

    new_upload = False

    for uploaded_file in uploaded_files:

        save_path = os.path.join(
            "uploads",
            uploaded_file.name
        )

        # Save only if file is new

        if not os.path.exists(save_path):

            with open(save_path, "wb") as f:

                f.write(uploaded_file.read())

            new_upload = True

    # Refresh vector DB only for new uploads

    if new_upload:

        if os.path.exists(VECTOR_DB_PATH):

            shutil.rmtree(VECTOR_DB_PATH)

        st.cache_resource.clear()

        st.success(
            "New PDFs uploaded and vector database refreshed!"
        )

    else:

        st.info(
            "Using existing vector database."
        )

# =========================
# DOCUMENT FILTER
# =========================

upload_folder = "uploads"

available_pdfs = []

if os.path.exists(upload_folder):

    available_pdfs = [
        file
        for file in os.listdir(upload_folder)
        if file.endswith(".pdf")
    ]

selected_pdf = st.selectbox(
    "Select Document",
    ["All Documents"] + available_pdfs
)

# =========================
# EMBEDDINGS
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# VECTORSTORE
# =========================

@st.cache_resource
def load_vectorstore():

    # Load existing vector DB

    if os.path.exists(VECTOR_DB_PATH):

        vectorstore = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vectorstore

    # Create vector DB

    documents = []

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

# =========================
# RETRIEVER WITH FILTER
# =========================

if selected_pdf == "All Documents":

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

else:

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {
                "source": selected_pdf
            }
        }
    )

# =========================
# MEMORY
# =========================

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# =========================
# LLM
# =========================

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# =========================
# CONVERSATIONAL CHAIN
# =========================

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
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

        # Conversational Retrieval

        response = conversation_chain.invoke(
            {
                "question": question
            }
        )

        answer = response["answer"]

        source_docs = response[
            "source_documents"
        ]

        # Sources

        sources = list(
            set(
                [
                    f"{doc.metadata.get('source')} (Page {doc.metadata.get('page')})"
                    for doc in source_docs
                ]
            )
        )

        # Assistant Response

        with st.chat_message(
            "assistant"
        ):

            response_placeholder = st.empty()

            full_response = ""

            # Streaming Response

            for word in answer.split():

                full_response += word + " "

                response_placeholder.markdown(
                    full_response + "▌"
                )

            response_placeholder.markdown(
                full_response
            )

            # Sources

            st.subheader(
                "Sources"
            )

            for source in sources:

                st.write(source)

    # Store Assistant Response

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )