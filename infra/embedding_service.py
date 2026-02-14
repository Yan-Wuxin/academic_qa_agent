import os
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings

class EmbeddingService:
    def __init__(self):
        self._embedding_model = None
        self._dimension = None
        self._init_model()

    def _init_model(self):
        load_dotenv()
        api_key = os.getenv("DASHBOARD_API_KEY")
        self._embedding_model = DashScopeEmbeddings(
            dashscope_api_key=api_key,
            model="BAAI/bge-small-zh"
        )
        self._dimension = 1536

    def get_model(self):
        return self._embedding_model

    def get_dimension(self):
        return self._dimension