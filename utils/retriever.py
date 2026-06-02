from rank_bm25 import BM25Okapi


def hybrid_retrieve(
    query,
    vectorstore,
    documents,
    k=10
):

    # ==========================================
    # SEMANTIC SEARCH WITH SCORES
    # ==========================================

    try:

        semantic_results = (
            vectorstore.similarity_search_with_score(
                query,
                k=k
            )
        )

        semantic_docs = []

        for doc, score in semantic_results:

            # ==========================================
            # FILTER LOW RELEVANCE RESULTS
            # ==========================================

            if score < 1.5:

                semantic_docs.append(
                    doc
                )

    except:

        semantic_docs = (
            vectorstore.similarity_search(
                query,
                k=k
            )
        )

    # ==========================================
    # BM25 KEYWORD SEARCH
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
    # TOP BM25 DOCUMENTS
    # ==========================================

    top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]

    keyword_docs = [
        documents[i]
        for i in top_indices
    ]

    # ==========================================
    # METADATA BOOSTING
    # ==========================================

    boosted_docs = []

    query_lower = query.lower()

    for doc in keyword_docs:

        source = str(
            doc.metadata.get(
                "source",
                ""
            )
        ).lower()

        if any(
            word in source
            for word in query_lower.split()
        ):

            boosted_docs.append(
                doc
            )

    # ==========================================
    # COMBINE RESULTS
    # ==========================================

    combined_docs = (
        boosted_docs
        + semantic_docs
        + keyword_docs
    )

    # ==========================================
    # REMOVE DUPLICATES
    # ==========================================

    unique_docs = []

    seen = set()

    for doc in combined_docs:

        content = doc.page_content

        if content not in seen:

            seen.add(
                content
            )

            unique_docs.append(
                doc
            )

    # ==========================================
    # RETURN TOP RESULTS
    # ==========================================

    return unique_docs[:k]