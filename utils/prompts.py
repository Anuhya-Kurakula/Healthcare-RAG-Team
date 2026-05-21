def build_prompt(
    context,
    question
):

    return f"""
You are a professional healthcare AI assistant.

STRICT RULES:
- Answer ONLY from provided healthcare context
- Do NOT hallucinate
- Do NOT use outside knowledge

Context:
{context}

Question:
{question}

Answer:
"""