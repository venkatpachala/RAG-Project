def answer_question(question: str):
    query_vector = embedder.embed_texts([question])[0]
    retrieved = store.search(query_vector, top_k=3)

    prompt = build_prompt(question, retrieved)
    answer = llm.generate(prompt)

    citations = []

    for chunk in retrieved:
        citations.append({
            "source": chunk["metadata"]["source"],
            "page": chunk["metadata"]["page"],
            "chunk_id": chunk["chunk_id"],
            "highlight_text": chunk["text"][:300]
        })

    return {
        "answer": answer,
        "citations": citations
    }
