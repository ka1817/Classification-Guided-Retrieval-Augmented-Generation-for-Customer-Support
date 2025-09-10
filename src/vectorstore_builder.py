import os
import logging
from typing import Dict
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from src.data_preprocessing import DataSplitting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("VectorStoreBuilder")


class VectorStoreBuilder:
    def __init__(self, vectorstore_dir: str = "vectorstores"):
        self.vectorstore_dir = vectorstore_dir
        os.makedirs(self.vectorstore_dir, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings()
        logger.info(f"VectorStoreBuilder initialized. Directory: {self.vectorstore_dir}")

    def build_and_save(self) -> Dict[str, FAISS]:
        splitter = DataSplitting()
        domain_docs = splitter.chunk_by_domain()

        domain_vectorstores = {}
        for domain, docs in domain_docs.items():
            logger.info(f"Building vectorstore for domain: {domain} with {len(docs)} docs")
            vectorstore = FAISS.from_documents(docs, self.embeddings)
            domain_vectorstores[domain] = vectorstore

            save_path = os.path.join(self.vectorstore_dir, domain)
            vectorstore.save_local(save_path)
            logger.info(f"✅ Saved vectorstore for domain '{domain}' at {save_path}")

        return domain_vectorstores

    def load(self, domain: str) -> FAISS:
        load_path = os.path.join(self.vectorstore_dir, domain)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No vectorstore found for domain: {domain}")

        logger.info(f"Loading vectorstore from {load_path}")
        return FAISS.load_local(
            load_path, self.embeddings, allow_dangerous_deserialization=True
        )

