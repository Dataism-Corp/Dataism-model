import sys, yaml
from retriever import retrieve
from assemble_prompt import build_prompt

def load_cfg():
    with open("config.yaml","r") as f:
        return yaml.safe_load(f)

def main():
    if len(sys.argv) < 2:
        print("Usage: python query_rag.py \"your question here\"")
        sys.exit(1)
    question = sys.argv[1]
    hits = retrieve(question)
    if not hits:
        print("No relevant context found in the KB.")
        sys.exit(0)

    prompt = build_prompt(question, hits)
    print("\n=== RAG PROMPT (send this to Qwen) ===\n")
    print(prompt)
    print("\n=== CONTEXT SOURCES ===")
    for h in hits:
        m = h["metadata"]
        title = m.get("title","Unknown")
        page = m.get("page",-1)
        sect = m.get("section","")
        print(f"- {title} | {'p.'+str(page) if page!=-1 else sect} | score={h.get('rerank_score',0):.3f}")

if __name__ == "__main__":
    main()
