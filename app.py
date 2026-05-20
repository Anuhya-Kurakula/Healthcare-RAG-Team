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

from langchain_community.vectorstores import (
    FAISS
)

from langchain_groq import (
    ChatGroq
)

from sentence_transformers import (
    CrossEncoder
)

# ==================================================
# LOAD ENV VARIABLES
# ==================================================

load_dotenv()

groq_api_key = os.getenv(
    "GROQ_API_KEY"
)

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Healthcare RAG Assistant",
    layout="wide"
)

# ==================================================
# TITLE
# ==================================================

st.title(
    "🏥 Healthcare RAG Assistant"
)

st.markdown(
    """
Ask healthcare-related questions based on uploaded healthcare documents using an advanced 6-stage RAG pipeline.
"""
)

# ==================================================
# EXAMPLE QUESTIONS
# ==================================================

with st.expander("💡 Example Questions"):

    st.markdown(
        """
- What are symptoms of dengue?
- How is diabetes diagnosed?
- Compare malaria and dengue symptoms.
- What treatment is recommended for hypertension?
"""
    )

# ==================================================
# EMBEDDING MODEL
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name="C:/models/all-MiniLM-L6-v2"
)

# ==================================================
# RERANKER MODEL
# ==================================================

reranker = CrossEncoder(
    "C:/hf_models/msmarco"
)

# ==================================================
# VECTOR DATABASE
# ==================================================

VECTOR_DB_PATH = "vectorstore"

@st.cache_resource
def load_vectorstore():

    # ==================================================
    # LOAD EXISTING VECTORSTORE
    # ==================================================

    if os.path.exists(VECTOR_DB_PATH):

        vectorstore = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vectorstore

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

    # ==================================================
    # CHUNKING
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
        embeddings
    )

    vectorstore.save_local(
        VECTOR_DB_PATH
    )

    return vectorstore

# ==================================================
# LOAD VECTORSTORE
# ==================================================

with st.spinner(
    "Loading Healthcare Knowledge Base..."
):

    vectorstore = load_vectorstore()

# ==================================================
# LOAD LLM
# ==================================================

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# ==================================================
# CONVERSATIONAL MEMORY
# ==================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

if "current_topic" not in st.session_state:

    st.session_state.current_topic = None

# ==================================================
# DISPLAY CHAT HISTORY
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ==================================================
# STAGE 1 — SMART QUERY REWRITING
# ==================================================

def rewrite_query(query):

    query = query.strip()

    query_lower = (
        query.lower()
        .replace("?", "")
        .replace(".", "")
        .strip()
    )

    # ==================================================
    # FOLLOW-UP QUERIES
    # ==================================================

    followup_queries = [
        "give symptoms",
        "symptoms",
        "give treatment",
        "treatment",
        "give curing methods",
        "causes",
        "complications",
        "how is it treated",
        "how is it diagnosed",
        "how is it managed",
        "how is it prevented",
        "what are symptoms",
        "what are complications",
        "what is the treatment",
        "how is treatment done"
    ]

    # ==================================================
    # COMPLETE QUESTIONS
    # ==================================================

    complete_question_starters = [
        "what",
        "who",
        "when",
        "where",
        "which",
        "define",
        "describe",
        "compare",
        "difference"
    ]

    words = query_lower.split()

    # ==================================================
    # DO NOT REWRITE COMPLETE QUESTIONS
    # ==================================================

    if (
        len(words) >= 3
        and
        words[0] in complete_question_starters
    ):

        return query

    # ==================================================
    # FOLLOW-UP DETECTION
    # ==================================================

    followup_detected = False

    for phrase in followup_queries:

        if phrase in query_lower:

            followup_detected = True

            break

    if not followup_detected:

        return query

    # ==================================================
    # USE CURRENT TOPIC
    # ==================================================

    last_topic = st.session_state.current_topic

    if not last_topic:

        return query

    # ==================================================
    # QUERY REWRITE PROMPT
    # ==================================================

    rewrite_prompt = f"""
You are a healthcare query rewriter.

Convert the follow-up healthcare query
into a complete standalone healthcare query.

IMPORTANT RULES:
- Use previous healthcare topic
- Keep query concise
- Do NOT hallucinate
- Return ONLY rewritten query

Previous Topic:
{last_topic}

Follow-up Query:
{query}

Rewritten Query:
"""

    rewritten_response = llm.invoke(
        rewrite_prompt
    )

    rewritten_query = (
        rewritten_response.content.strip()
    )

    return rewritten_query

# ==================================================
# TOPIC EXTRACTION
# ==================================================

def extract_topic(query):

    prompt = f"""
Extract ONLY the main healthcare topic
from this query.

Examples:
Query: What is malaria?
Topic: malaria

Query: Symptoms of diabetes
Topic: diabetes

Query: How is dengue treated?
Topic: dengue

Return ONLY topic name.

Query:
{query}

Topic:
"""

    response = llm.invoke(
        prompt
    )

    topic = response.content.strip().lower()

    return topic

# ==================================================
# STAGE 3 — RERANKING
# ==================================================

def rerank_documents(query, docs):

    pairs = [
        (query, doc.page_content)
        for doc in docs
    ]

    scores = reranker.predict(
        pairs
    )

    # ==================================================
    # SORT USING SCORE ONLY
    # ==================================================

    ranked_docs = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    return [
        doc
        for score, doc in ranked_docs
    ]

# ==================================================
# STAGE 4 — REFINE DOCUMENTS
# ==================================================

def refine_documents(docs):

    unique_docs = []

    seen = set()

    for doc in docs:

        text = doc.page_content.strip()

        if text not in seen:

            unique_docs.append(doc)

            seen.add(text)

    return unique_docs

# ==================================================
# CHAT INPUT
# ==================================================

question = st.chat_input(
    "Ask a healthcare question..."
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
        # STAGE 1 — QUERY REWRITE
        # ==================================================

        rewritten_query = rewrite_query(
            question
        )

        # ==================================================
        # STORE CURRENT TOPIC
        # ==================================================

        new_topic = extract_topic(
            rewritten_query
        )

        if (
            new_topic
            and
            len(new_topic.split()) <= 4
        ):

            st.session_state.current_topic = new_topic

        # ==================================================
        # STAGE 2 — RETRIEVAL
        # ==================================================

        retrieved_docs_with_scores = (
            vectorstore.similarity_search_with_score(
                rewritten_query,
                k=5
            )
        )

        # ==================================================
        # SIMILARITY FILTER
        # ==================================================

        SIMILARITY_THRESHOLD = 15

        retrieved_docs = []

        for doc, score in retrieved_docs_with_scores:

            if score <= SIMILARITY_THRESHOLD:

                retrieved_docs.append(doc)

        # ==================================================
        # HANDLE NO DOCUMENTS
        # ==================================================

        if not retrieved_docs:

            response_text = (
                "The information is not available "
                "in the uploaded healthcare documents."
            )

            sources = []

        else:

            # ==================================================
            # STAGE 3 — RERANKING
            # ==================================================

            reranked_docs = rerank_documents(
                rewritten_query,
                retrieved_docs
            )

            # ==================================================
            # STAGE 4 — REFINEMENT
            # ==================================================

            refined_docs = refine_documents(
                reranked_docs
            )

            # ==================================================
            # STAGE 5 — CONTEXT CREATION
            # ==================================================

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in refined_docs
                ]
            )

            # ==================================================
            # SOURCES
            # ==================================================

            sources = list(
                set(
                    [
                        f"{doc.metadata.get('source')} "
                        f"(Page {doc.metadata.get('page')})"
                        for doc in refined_docs
                    ]
                )
            )

            # ==================================================
            # FINAL PROMPT
            # ==================================================

            prompt = f"""
You are a professional healthcare AI assistant.

STRICT RULES:
- Answer ONLY from provided healthcare context
- Do NOT use outside knowledge
- Do NOT hallucinate
- Do NOT guess
- If answer unavailable, say EXACTLY:
The information is not available in the uploaded healthcare documents.

Formatting Rules:
- Use bullet points where needed
- Keep answers concise
- Keep answers readable

Context:
{context}

Question:
{question}

Answer:
"""

            # ==================================================
            # STAGE 6 — GENERATION
            # ==================================================

            response = llm.invoke(
                prompt
            )

            response_text = response.content

            # ==================================================
            # CLEAN RESPONSE
            # ==================================================

            response_text = (
                response_text
                .replace("z ", "• ")
                .replace("\\n", "\n")
            )

            # ==================================================
            # REMOVE SOURCES IF INVALID
            # ==================================================

            if (
                "not available" in response_text.lower()
                or
                "not found" in response_text.lower()
            ):

                sources = []

    # ==================================================
    # STORE ASSISTANT RESPONSE
    # ==================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text
        }
    )

    # ==================================================
    # DISPLAY ASSISTANT RESPONSE
    # ==================================================

    with st.chat_message("assistant"):

        st.write(response_text)

        # ==================================================
        # SHOW REWRITTEN QUERY
        # ==================================================

        if rewritten_query.lower() != question.lower():

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

                    st.write(
                        f"• {source}"
                    )