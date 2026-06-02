from sklearn.metrics.pairwise import (
    cosine_similarity
)


def rerank_documents(
    query,
    docs,
    embeddings
):

    # ==========================================
    # QUERY EMBEDDING
    # ==========================================

    query_embedding = embeddings.embed_query(
        query
    )

    doc_scores = []

    # ==========================================
    # COMPUTE SIMILARITY
    # ==========================================

    for doc in docs:

        doc_embedding = embeddings.embed_query(
            doc.page_content
        )

        score = cosine_similarity(
            [query_embedding],
            [doc_embedding]
        )[0][0]

        # ==========================================
        # FILTER LOW QUALITY DOCUMENTS
        # ==========================================

        if score > 0.20:

            doc_scores.append(
                (
                    score,
                    doc
                )
            )

    # ==========================================
    # SORT BY SCORE
    # ==========================================

    ranked_docs = sorted(
        doc_scores,
        key=lambda x: x[0],
        reverse=True
    )

    # ==========================================
    # RETURN TOP DOCUMENTS
    # ==========================================

    return [
        doc
        for score, doc in ranked_docs[:8]
    ]