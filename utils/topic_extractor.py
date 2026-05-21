def extract_topic(
    query,
    llm
):

    prompt = f"""
Extract ONLY the main healthcare topic.

Query:
{query}

Topic:
"""

    response = llm.invoke(
        prompt
    )

    return (
        response.content
        .strip()
        .lower()
    )