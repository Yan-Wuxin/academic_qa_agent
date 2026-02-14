from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_document(file_path):
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=['\n', '\n\n', '.', '!', '?', ', ', ' '],
        length_function=len
    )
    splits = splitter.split_documents(docs)
    return splits