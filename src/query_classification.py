import os
import pandas as pd
import logging
import joblib
from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from src.data_ingestion import DataIngestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("Classification")


class Classification:
    def __init__(self, test_size: float = 0.2, random_state: int = 42, model_path: str = "models/classifier_pipeline.pkl"):
        self.test_size = test_size
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self.model_path = model_path
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def load_and_split_data(self) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        try:
            data = DataIngestion()
            df = data.load_data()
            X = df['query']
            y = df['domain']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            logger.info(f"Data split into train and test sets. Train size: {len(X_train)}, Test size: {len(X_test)}")
            return X_train, X_test, y_train, y_test
        except KeyError as e:
            logger.error(f"Missing expected column in dataset: {e}")
            raise
        except Exception as e:
            logger.exception(f"Error during data loading and splitting: {e}")
            raise

    def build_pipeline(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', MultinomialNB())
        ])
        logger.info("Classification pipeline created successfully.")
        return self.pipeline

    def train_and_evaluate(self):
        try:
            X_train, X_test, y_train, y_test = self.load_and_split_data()
            self.build_pipeline()
            self.pipeline.fit(X_train, y_train)
            logger.info("Model training completed.")
            y_pred = self.pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Classification Accuracy: {accuracy:.4f}")
            logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
            joblib.dump(self.pipeline, self.model_path)
            logger.info(f"Pipeline saved successfully at {self.model_path}")

        except Exception as e:
            logger.exception(f"Error during training or evaluation: {e}")
            raise

    def load_pipeline(self):
        try:
            if os.path.exists(self.model_path):
                self.pipeline = joblib.load(self.model_path)
                logger.info(f"Pipeline loaded from {self.model_path}")
            else:
                raise FileNotFoundError(f"No pipeline found at {self.model_path}")
        except Exception as e:
            logger.exception(f"Error loading pipeline: {e}")
            raise


