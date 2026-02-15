from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config_handler import chroma_conf

class SplitterService:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len
        )

    def split_document(self, document):
        return self.splitter.split_documents(document)