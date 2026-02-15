from langchain_core.runnables import RunnableLambda

from rag.reranker import Reranker
from rag.retriever_builder import build_retriever
from infra.loader_service import LoaderService
from infra.splitter_service import SplitterService
from rag.rag_pipeline import RAGPipeline
from core.memory import add_memory_to_chain
from model.factory import chat_model, embed_model

def rag_qa(file_path, input, session_id):
    docs = LoaderService().load_file(file_path)
    splits = SplitterService().split_document(docs)

    retriever = build_retriever(splits)
    pipeline = RAGPipeline(
        retriever=retriever,
        llm=chat_model,
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
