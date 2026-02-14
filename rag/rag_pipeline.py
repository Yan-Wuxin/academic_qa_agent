from langchain_core.prompts import PromptTemplate
from utils.prompt_loader import load_rag_prompt
from rag.query_transform import build_query_rewrite_chain, build_hyde_chain

class RAGPipeline:
    def __init__(self, retriever, llm, reranker=None):
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker

        self.rewrite_chain = build_query_rewrite_chain(llm)
        self.hyde_chain = build_hyde_chain(llm)

        self.prompt = PromptTemplate.from_template(load_rag_prompt())

    def rewrite(self, query):
        return self.rewrite_chain.invoke({"query": query})

    def hyde(self, query):
        return self.hyde_chain.invoke({"query": query})

    def retrieve(self, query):
        return self.retriever.invoke(query)

    def rerank_docs(self, query, docs):
        if self.reranker:
            return self.reranker.rerank(query, docs)
        return docs

    def format_docs(self, docs):
        if not docs:
            return "无参考资料"
        return "\n\n".join(doc.page_content for doc in docs)

    def run(self, query, chat_history=None):
        rewritten = self.rewrite(query)
        hyde_query = self.hyde(rewritten)

        docs = self.retrieve(hyde_query)
        docs = self.rerank_docs(rewritten, docs)

        context = self.format_docs(docs)

        final_prompt = self.prompt.invoke({
            "input": query,
            "context": context,
            "chat_history": chat_history or []
        })

        return self.llm.invoke(final_prompt)

