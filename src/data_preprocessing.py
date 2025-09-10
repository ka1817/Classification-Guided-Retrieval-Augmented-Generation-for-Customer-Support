import logging
from langchain.schema import Document
from src.data_ingestion import DataIngestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("DataPreprocessing")


class DataSplitting:
    def __init__(self):
        logger.info("Initialized DataSplitting")

    def chunking(self):
        logger.info("Starting document ingestion...")
        data = DataIngestion()
        df = data.load_data()
        logger.info(f"Received data with shape {df.shape} for creating documents.")

        chunks = []
        for _, row in df.iterrows():
            doc = Document(
                page_content=f"Query: {row['query']}\nResponse: {row['response']}\nIntent: {row['intent']}",
                metadata={"domain": row["domain"], "intent": row["intent"]},
            )
            chunks.append(doc)

        logger.info(f"Converted {len(chunks)} rows into Document objects.")
        return chunks

    def chunk_by_domain(self):
        data = DataIngestion()
        df = data.load_data()

        domain_chunks = {}
        for domain in df["domain"].unique():
            domain_docs = [
                Document(
                    page_content=f"Query: {row['query']}\nResponse: {row['response']}\nIntent: {row['intent']}",
                    metadata={"domain": row["domain"], "intent": row["intent"]},
                )
                for _, row in df[df["domain"] == domain].iterrows()
            ]
            domain_chunks[domain] = domain_docs
            logger.info(f"Created {len(domain_docs)} documents for domain '{domain}'.")

        return domain_chunks
