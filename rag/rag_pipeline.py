from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from utils.prompt_loader import load_rag_prompt
from rag.query_transform import build_query_rewrite_chain, build_hyde_chain
from core.rag import llm

def build_rag_chain(retriever, llm):
    rewrite_chain = build_query_rewrite_chain(llm)
    hyde_chain = build_hyde_chain(llm)
    def rewrite_query(inputs):
        return rewrite_chain.invoke({"query": inputs["input"]})
    def hyde_query(query):
        return hyde_chain.invoke({"query": query})
    def format_func(docs: list[Document]):
        if not docs:
            return "无参考资料"
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = PromptTemplate.from_template(
        load_rag_prompt()
    )
    rag_chain = {"input": lambda x: x["input"],
                 "context": lambda x:
                    retriever.invoke(
                        hyde_query(
                            rewrite_query(x)
                        )
                    )
                    | format_func
                } \
                | prompt \
                | llm \
                | StrOutputParser()
    return rag_chain

# test
if __name__ == "__main__":
    rewrite_chain = build_query_rewrite_chain(llm)
    hyde_chain = build_hyde_chain(llm)
    input = "残差网络是什么东西？"
    chain = (rewrite_chain | RunnableLambda(lambda query: {"query": query}) | hyde_chain)
    response = chain.invoke({"query": input})
    print(response)