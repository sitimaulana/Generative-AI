# 10.9 Module 10 Exercises (Halaman 124-125)
# 1. ask(question: str) wrapper returning answer and source files
def ask_with_sources(query_engine, question: str) -> dict:
    resp = query_engine.query(question)
    sources = [node.metadata.get("source", "unknown") for node in resp.source_nodes]
    return {"answer": str(resp), "sources": list(set(sources))}

# 2. Re-rank nodes with LLM score
def rerank_nodes(nodes, question: str, score_fn, top_n: int = 3):
    scored = [(n, score_fn(question, n.text)) for n in nodes]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [n for n, s in scored[:top_n]]

if __name__ == "__main__":
    print("Module 10 Exercises utilities ready.")
