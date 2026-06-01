import streamlit as st

from utils.embeddings import (
    load_embeddings
)

from utils.vectorstore import (
    load_vectorstore
)

from utils.llm import (
    load_llm
)

from utils.memory import (
    initialize_memory
)

from utils.query_rewriter import (
    rewrite_query
)

from utils.topic_extractor import (
    extract_topic
)

from utils.retriever import (
    hybrid_retrieve
)

from utils.reranker import (
    rerank_documents
)

from utils.prompts import (
    build_prompt
)

from langchain_community.document_loaders import (
    PyPDFLoader
)

import os

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Healthcare RAG Assistant",
    layout="wide"
)

# ==================================================
# CUSTOM UI
# ==================================================

st.markdown(
    """
<style>

.main {
    background-color: #f8fafc;
}

.stChatMessage {
    border-radius: 15px;
    padding: 10px;
}

h1 {
    color: #0f172a;
}

</style>
""",
    unsafe_allow_html=True
)

# ==================================================
# TITLE
# ==================================================

st.title(
    "🏥 Healthcare RAG Assistant"
)

st.markdown(
    """
Ask healthcare-related questions based on uploaded healthcare documents using an advanced multi-stage RAG pipeline.
"""
)

# ==================================================
# EXAMPLE QUESTIONS
# ==================================================

with st.expander("💡 Example Questions"):

    st.markdown(
        """
- What are symptoms of dengue?
- How is malaria treated?
- Compare malaria and dengue symptoms.
- What are causes of depression?
"""
    )

# ==================================================
# INITIALIZE MEMORY
# ==================================================

initialize_memory()

# ==================================================
# LOAD COMPONENTS
# ==================================================

embeddings = load_embeddings()

vectorstore = load_vectorstore(
    embeddings
)

llm = load_llm()

# ==================================================
# LOAD DOCUMENTS FOR HYBRID SEARCH
# ==================================================

documents = []

upload_folder = "uploads"

if os.path.exists(upload_folder):

    pdf_files = [
        file
        for file in os.listdir(upload_folder)
        if file.endswith(".pdf")
    ]

    for file in pdf_files:

        loader = PyPDFLoader(
            os.path.join(
                upload_folder,
                file
            )
        )

        docs = loader.load()

        for doc in docs:

            doc.metadata["source"] = file

        documents.extend(docs)

# ==================================================
# DISPLAY CHAT HISTORY
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

# ==================================================
# CHAT INPUT
# ==================================================

question = st.chat_input(
    "Ask healthcare question..."
)

# ==================================================
# MAIN PIPELINE
# ==================================================

if question:

    # ==================================================
    # STORE USER MESSAGE
    # ==================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # ==================================================
    # DISPLAY USER MESSAGE
    # ==================================================

    with st.chat_message("user"):

        st.write(question)

    with st.spinner(
        "Generating answer..."
    ):

        # ==================================================
        # QUERY REWRITE
        # ==================================================

        rewritten_query = rewrite_query(
            question,
            llm
        )

        # ==================================================
        # TOPIC EXTRACTION
        # ==================================================

        topic = extract_topic(
            rewritten_query,
            llm
        )

        if (
            topic
            and
            len(topic.split()) <= 4
        ):

            st.session_state.current_topic = topic

        # ==================================================
        # HYBRID RETRIEVAL
        # ==================================================

        retrieved_docs = hybrid_retrieve(
            rewritten_query,
            vectorstore,
            documents
        )

        # ==================================================
        # HANDLE NO DOCS
        # ==================================================

        if not retrieved_docs:

            answer = (
                "The information is not available "
                "in the uploaded healthcare documents."
            )

            sources = []

        else:

            # ==================================================
            # RERANKING
            # ==================================================

            reranked_docs = rerank_documents(
                rewritten_query,
                retrieved_docs,
                embeddings
            )

            # ==================================================
            # CONTEXT
            # ==================================================

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in reranked_docs
                ]
            )

            # ==================================================
            # SOURCES
            # ==================================================

            sources = sorted(
                list(
                    set(
                        [
                            f"{doc.metadata.get('source')} "
                            f"(Page {doc.metadata.get('page')})"
                            for doc in reranked_docs
                            if doc.metadata.get("source")
                        ]
                    )
                )
            )

            # ==================================================
            # PROMPT
            # ==================================================

            prompt = build_prompt(
                context,
                rewritten_query
            )

            # ==================================================
            # GENERATE RESPONSE
            # ==================================================

            response = llm.invoke(
                prompt
            )

            answer = response.content

            answer = (
                answer.replace(
                    "\\n",
                    "\n"
                )
            )

            # ==================================================
            # REMOVE SOURCES IF INVALID
            # ==================================================

            if any(
                phrase in answer.lower()
                for phrase in [
                    "not available",
                    "not found",
                    "not mention",
                    "not aware",
                    "outside the provided context",
                    "no information",
                    "unable to provide",
                    "there is no information",
                    "no relevant data available",
                    "does not contain any information",
                    "not related to the healthcare context",
                    "cannot provide information"
                ]
            ):
                sources = []

    # ==================================================
    # STORE ASSISTANT RESPONSE
    # ==================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # ==================================================
    # DISPLAY RESPONSE
    # ==================================================

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            answer
        )

        # ==================================================
        # SHOW REWRITTEN QUERY
        # ==================================================

        with st.expander(
            "🔄 View Rewritten Query"
        ):

            st.write(
                rewritten_query
            )

        # ==================================================
        # SHOW SOURCES
        # ==================================================

        if sources:

            with st.expander(
                "📚 View Sources"
            ):

                for source in sources:

                    st.markdown(
                        f"• {source}"
                    )