import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader

class LoaderService:
    def __init__(self):
        self.loader_registry = {
            ".pdf": self._load_pdf,
            ".txt": self._load_txt
        }

    def load_file(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.loader_registry:
            raise FileNotFoundError(f"Unsupported file type: {ext}")
        return self.loader_registry[ext](file_path)

    def _load_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        return loader.load()

    def _load_txt(self, file_path):
        loader = TextLoader(file_path)
        return loader.load()

    def load_directory(self, directory):
        all_docs = []
        for root, _, files in os.walk(directory): # 目录批量加载
            for file in files:
                path = os.path.join(root, file)
                try:
                    docs = self.load_file(path)
                    all_docs.append(docs)
                except Exception:
                    pass
        return all_docs