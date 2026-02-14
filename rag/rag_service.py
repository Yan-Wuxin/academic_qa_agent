import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings

from rag.reranker import Reranker
from rag.retriever_builder import build_retriever
from infra.loader_service import load_and_split_document
from rag.rag_pipeline import RAGPipeline
from core.memory import add_memory_to_chain

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
# initialize the model and embedding
llm = ChatTongyi(
    dashscope_api_key=api_key,
    model_name='qwen-turbo',
    # temperature=
)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def rag_qa(file_path, input, session_id):
    docs = load_and_split_document(file_path)
    retriever = build_retriever(docs, embeddings)
    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        reranker=Reranker()
    )
    runnable_pipeline = RunnableLambda(
        lambda inputs: pipeline.run(
            inputs["input"],
            inputs.get("chat_history")
        )
    )
    rag_chain = add_memory_to_chain(runnable_pipeline, session_id)
    response = rag_chain.invoke(
        {"input": input}, # 需传入字典
        config={"configurable": {"session_id": session_id}},
    )
    return response

if __name__ == "__main__":
    try:
        print(embeddings.embed_query("hello"))
    except Exception as e:
        print("error:", e)