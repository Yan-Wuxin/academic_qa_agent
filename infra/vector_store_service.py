from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from model.factory import embed_model

class VectorStoreService:

    def __init__(self):
        self.embedding_model = embed_model
        self.vector_store = self._init_store()

    def _init_store(self):
        return Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=self.embedding_model,
            persist_directory=chroma_conf["persist_directory"],
        )

    def add_documents(self, documents):
        self.vector_store.add_documents(documents)

    def get_retriever(self, k=None):
        k = k or chroma_conf["k"]
        return self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )

    def reset_collection(self):
        self.vector_store.delete_collection()
        self.vector_store = self._init_store()