import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models.tongyi import ChatTongyi

from .memory import add_memory_to_chain ###


load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
# initialize the model and embedding
llm = ChatTongyi(
    dashscope_api_key=api_key,
    model_name='qwen-turbo',
    # temperature=
)
embeddings = DashScopeEmbeddings()

def load_and_split_document(file_path):
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        separators=['\n', '\n\n', '。', '！', '？', '，', '、', ' '],
        length_function=len
    )
    splits = splitter.split_documents(docs)
    return splits

def build_vector_db(splits, persist_directory="./chroma_db"):
    vector_db = Chroma(
        collection_name="test",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    vector_db.add_documents(splits)
    retriever = vector_db.as_retriever(search_kwargs={"k":5})
    return retriever

def build_rag_chain(retriever):
    def format_func(docs: list[Document]):
        if not docs:
            return "无参考资料"
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = PromptTemplate.from_template(
        """
        你是一个学术文献问答助手，请遵循严谨求实的态度，严格基于提供的文献内容回答问题，不要自行编造信息，
        若文献中没有相关内容，请明确说明”未在文献中找到相关答案“
        
        文献内容：
        {context}
        
        用户问题：
        {input}
        
        请给出清晰、准确、严谨的回答：
        """
    )
    rag_chain = {"input": RunnableLambda(itemgetter("input")),
                 "context": RunnableLambda(itemgetter("input")) | retriever | format_func} \
            | prompt \
            | llm \
            | StrOutputParser()
    return rag_chain

def rag_qa(file_path, input, session_id="default_session"):
    docs = load_and_split_document(file_path)
    retriever = build_vector_db(docs)
    chain = build_rag_chain(retriever)
    rag_chain = add_memory_to_chain(chain, session_id)
    response = rag_chain.invoke(
        {"input": input}, # 需传入字典
        config={"configurable": {"session_id": session_id}},
    )
    return response