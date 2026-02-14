from evaluation.retriever_eval import load_testset, evaluate_recall
from rag.retriever_builder import build_retriever
from langchain_huggingface import HuggingFaceEmbeddings
from infra.loader_service import load_and_split_document

from rag.hybrid_retriever import HybridRetriever

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def main():
    docs = load_and_split_document("D:/academic_qa_agent/data/2210.03629v3.pdf")
    # add metadata
    splits = []
    for idx, doc in enumerate(docs):
        doc.metadata["chunk_id"] = f"react_paper_{idx}"
        # print(doc.page_content[:200])
        splits.append(doc)

    vector_retriever, bm25_retriever = build_retriever(splits, embeddings)
    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
    )
    testset = load_testset()
    print("vector_retriever")
    evaluate_recall(retriever=vector_retriever, testset=testset)
    print("\nretriever")
    evaluate_recall(retriever=retriever, testset=testset)

if __name__ == "__main__":
    main()


