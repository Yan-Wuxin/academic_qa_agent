import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

from rag.retriever_builder import build_retriever
from infra.loader_service import load_and_split_document
from rag.rag_pipeline import build_rag_chain
from core.memory import add_memory_to_chain

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
# initialize the model and embedding
llm = ChatTongyi(
    dashscope_api_key=api_key,
    model_name='qwen-turbo',
    # temperature=
)
embeddings = DashScopeEmbeddings(
    dashscope_api_key=api_key,
    model="text-embedding-v3",
)

def rag_qa(file_path, input, session_id):
    docs = load_and_split_document(file_path)
    retriever = build_retriever(docs, embeddings)
    chain = build_rag_chain(retriever, llm)
    rag_chain = add_memory_to_chain(chain, session_id)
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