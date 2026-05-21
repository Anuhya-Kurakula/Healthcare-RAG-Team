def retrieve_documents(
    query,
    vectorstore
):

    retrieved = (
        vectorstore.similarity_search_with_score(
            query,
            k=5
        )
    )

    threshold = 15

    docs = []

    for doc, score in retrieved:

        if score <= threshold:

            docs.append(doc)

    return docs