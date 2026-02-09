import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models.tongyi import ChatTongyi

from memory import add_memory_to_chain


load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
# initialize the model and embedding
llm = ChatTongyi(
    dashscope_api_key=api_key,
    model_name='qwen-turbo',
    # temperature=
)
embeddings = HuggingFaceEmbeddings(
    model_name='qwen-turbo'
)

def load_and_split_document(file_path):
    loader = PyPDFLoader(
        file_path=file_path,
        mode='page',
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator=['\n', '\n\n', '。', '！', '？', '，', '、', ' '],
        length_function=len
    )
    splits = splitter.split_documents(docs)
    return splits

def build_vector_db(splits, persist_directory="./"):
    vector_db = Chroma.from_documents(
        documents=splits,
        embeddings=embeddings,
        persist_directory=persist_directory,
    )
    retriever = vector_db.as_retriever(search_kwargs={"k":5})
    return retriever

def build_rag_chain(retriever):
    def format_func(docs: list[Document]):
        if not docs:
            return "无参考资料"
        formatted_docs = "["
        for doc in docs:
            formatted_docs += doc.page_content
        formatted_docs += "]"
        return formatted_docs

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
    rag_chain = {"input": RunnablePassthrough(), "content": retriever | format_func()} \
            | prompt \
            | llm \
            | StrOutputParser()
    return rag_chain

def rag_qa(file_path, input, session_id="default_session"):
    docs = load_and_split_document(file_path)
    retriever = build_vector_db(docs)
    chain = build_rag_chain(retriever)
    rag_chain = add_memory_to_chain(chain, session_id)
    response = rag_chain(
        input,
        config={"configurable": {"session_id": session_id}},
    )
    return response