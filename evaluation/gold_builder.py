import json
import os

TESTSET_PATH = "testset1.json"

def load_testset():
    if not os.path.exists(TESTSET_PATH):
        return []
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_testset(testset):
    with open(TESTSET_PATH, "w", encoding="utf-8") as f:
        json.dump(testset, f, ensure_ascii=False, indent=2)

def build_gold_entry(retriver, query, k=10):
    docs = retriver.invoke(query)

    print("\n==========================")
    print("Query:", query)
    print("==========================\n")

    for i, doc in enumerate(docs[:k]):
        id = doc.metadata["chunk_id"]
        print(f"{i} chunk_id: {id}")
        print(doc.page_content[:400])
        print("-"*20)

    print("\n请输入相关chunk编号（如：0 2 4）")
    user_input = input(">>>")
    indices = [int(x) for x in user_input.split(" ")]

    selected_chunk = []
    selected_texts = []

    for idx in indices:
        doc = docs[idx]
        selected_chunk.append(doc.metadata.get("chunk_id"))
        selected_texts.append(doc.page_content)

    return {
        "query": query,
        "gold_chunk": selected_chunk,
        "gold_texts": selected_texts,
    }

def interactive_builder(retriever):
    testset = load_testset()
    print("\n请输入测试问题（键入exit退出）：")
    while True:
        query = input(">>> ")
        if query == "exit":
            break
        entry = build_gold_entry(retriever, query)
        testset.append(entry)
        save_testset(testset)
        print("已保存\n")

if __name__ == "__main__":
    from rag.retriever_builder import build_retriever
    from rag.hybrid_retriever import HybridRetriever
    from evaluation.run_eval import embeddings
    from infra.loader_service import load_and_split_document

    docs = load_and_split_document("D:/academic_qa_agent/data/2303.11366v4.pdf")
    splits = []
    for idx, doc in enumerate(docs):
        doc.metadata["chunk_id"] = f"paper_{idx}"
        splits.append(doc)
    vector_retriever, bm25_retriever = build_retriever(splits, embeddings)
    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
    )
    interactive_builder(retriever)