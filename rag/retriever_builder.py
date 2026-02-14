from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from rag.hybrid_retriever import HybridRetriever

def build_retriever(splits,
                    embeddings,
                    k=5,
                    persist_directory="./chroma_db",
                    collection_name="test"):
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 5

    vector_db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    vector_db.add_documents(splits)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k":k})

    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
    )
    # return retriever
    return vector_retriever, bm25_retriever