from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model="BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model)

    def rerank(self, query, docs, top_k=5):
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.model.predict(pairs)
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]