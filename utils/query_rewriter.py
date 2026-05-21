import streamlit as st


def rewrite_query(
    query,
    llm
):

    # ==================================================
    # CLEAN QUERY
    # ==================================================

    query = query.strip()

    query_lower = (
        query.lower()
        .replace("?", "")
        .replace(".", "")
        .strip()
    )

    # ==================================================
    # FOLLOW-UP QUERY PATTERNS
    # ==================================================

    followup_queries = [

        "symptoms",
        "give symptoms",
        "what are symptoms",

        "treatment",
        "give treatment",
        "how is it treated",
        "how it is treated",

        "diagnosis",
        "how is it diagnosed",

        "prevention",
        "how is it prevented",

        "causes",
        "complications",
        "side effects",

        "management",
        "how is it managed"
    ]

    # ==================================================
    # DETECT FOLLOW-UP
    # ==================================================

    followup_detected = any(
        phrase in query_lower
        for phrase in followup_queries
    )

    # ==================================================
    # RETURN ORIGINAL IF NOT FOLLOW-UP
    # ==================================================

    if not followup_detected:

        return query

    # ==================================================
    # GET LAST TOPIC
    # ==================================================

    last_topic = st.session_state.current_topic

    if not last_topic:

        return query

    # ==================================================
    # STRICT REWRITE PROMPT
    # ==================================================

    rewrite_prompt = f"""
You are a healthcare query rewriter.

Your task:
Convert follow-up healthcare questions into standalone questions.

STRICT RULES:
- Return ONLY the rewritten question
- Do NOT explain
- Do NOT answer
- Do NOT add bullet points
- Do NOT add extra text
- Output must contain ONLY ONE sentence

Examples:

Previous Topic: malaria
Follow-up: give symptoms
Output: What are the symptoms of malaria?

Previous Topic: dengue
Follow-up: how is it treated
Output: How is dengue treated?

Previous Topic:
{last_topic}

Follow-up:
{query}

Output:
"""

    # ==================================================
    # LLM CALL
    # ==================================================

    response = llm.invoke(
        rewrite_prompt
    )

    # ==================================================
    # CLEAN OUTPUT
    # ==================================================

    rewritten_query = (
        response.content.strip()
    )

    rewritten_query = (
        rewritten_query.split("\n")[0]
    )

    # ==================================================
    # FALLBACK
    # ==================================================

    if not rewritten_query:

        return query

    return rewritten_query