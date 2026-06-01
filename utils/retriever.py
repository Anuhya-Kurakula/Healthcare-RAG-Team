from rank_bm25 import BM25Okapi


def hybrid_retrieve(
    query,
    vectorstore,
    documents,
    k=10
):

    # ==========================================
    # SEMANTIC SEARCH
    # ==========================================

    semantic_docs = vectorstore.similarity_search(
        query,
        k=10
    )

    # ==========================================
    # KEYWORD SEARCH (BM25)
    # ==========================================

    tokenized_docs = [
        doc.page_content.lower().split()
        for doc in documents
    ]

    bm25 = BM25Okapi(
        tokenized_docs
    )

    tokenized_query = (
        query.lower().split()
    )

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    # ==========================================
    # TOP KEYWORD DOCUMENTS
    # ==========================================

    top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:10]

    keyword_docs = [
        documents[i]
        for i in top_indices
    ]

    # ==========================================
    # COMBINE RESULTS
    # ==========================================

    combined_docs = (
        semantic_docs +
        keyword_docs
    )

    # ==========================================
    # REMOVE DUPLICATES
    # ==========================================

    unique_docs = []

    seen = set()

    for doc in combined_docs:

        content = doc.page_content

        if content not in seen:

            seen.add(content)

            unique_docs.append(doc)

    return unique_docs[:10]