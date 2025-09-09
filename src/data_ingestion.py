import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataIngestion")

class DataIngestion:
    def __init__(self, path: str = None):
        if path:
            self.path = path
        else:
            default_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "domain_specific_chatbot_data (1).csv"
            )
            self.path = os.path.abspath(default_path)
            logger.info(f"No path provided. Using default file: {self.path}")

    def load_data(self) -> pd.DataFrame:
        try:
            if not os.path.exists(self.path):
                logger.error(f"File not found at {self.path}")
                raise FileNotFoundError(f"No file found at {self.path}")

            df = pd.read_csv(self.path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.exception(f"Error loading data: {e}")
            raise

if __name__ == "__main__":
    data_ingestor = DataIngestion()
    df = data_ingestor.load_data()
