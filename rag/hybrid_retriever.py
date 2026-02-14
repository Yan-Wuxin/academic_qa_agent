from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from collections import defaultdict

class HybridRetriever(BaseRetriever):
    # pydantic 规则
    vector_retriever: BaseRetriever
    bm25_retriever: BaseRetriever
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    k: int = 5

    def _get_relevant_documents(self, query: str) -> list[Document]:
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        score_map = defaultdict(float)
        doc_map = {}
        for rank, doc in enumerate(vector_docs):
            key = self._doc_key(doc)
            score_map[key] += self.vector_weight * (1 / (rank + 1))
            doc_map[key] = doc
        for rank, doc in enumerate(bm25_docs):
            key = self._doc_key(doc)
            score_map[key] += self.bm25_weight * (1 / (rank + 1))
            doc_map[key] = doc
        sorted_keys = sorted(score_map, key=score_map.get, reverse=True)
        return [doc_map[doc] for doc in sorted_keys[:self.k]]

    def _doc_key(self, doc: Document) -> str:
        return (
            doc.metadata.get("source"),
            doc.metadata.get("chunk_id")
        )