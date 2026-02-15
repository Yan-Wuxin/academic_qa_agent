from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from rag.hybrid_retriever import HybridRetriever
from infra.vector_store_service import VectorStoreService

def build_retriever(splits):
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 10

    vector_retriever = VectorStoreService().get_retriever()

    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
    )
    return retriever