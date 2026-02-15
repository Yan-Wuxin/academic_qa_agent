from abc import ABC, abstractmethod
from typing import Optional
import os
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import BaseChatModel, ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from utils.config_handler import rag_config

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass

class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        provider = rag_config["chat_model"]["provider"]
        model_name = rag_config["chat_model"]["model_name"]
        if provider == "dashscope":
            load_dotenv()
            api_key = os.getenv("DASHSCOPE_API_KEY")
            return ChatTongyi(
                model=model_name,
                api_key=api_key,
            )
        raise ValueError(f"Unknown model provider {provider}")

class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        provider = rag_config["embedding"]["provider"]
        model_name = rag_config["embedding"]["model_name"]
        if provider == "local":
            return HuggingFaceEmbeddings(model_name=model_name)
        if provider == "dashscope":
            return DashScopeEmbeddings(model=model_name)
        raise ValueError(f"Unknown embedding provider {provider}")

chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()

