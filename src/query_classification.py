import pandas as pd
import logging
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
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None

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
        """Build the classification pipeline."""
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', MultinomialNB())
        ])
        return self.pipeline
        logger.info("Classification pipeline created successfully.")

    def train_and_evaluate(self):
        """Train the pipeline and evaluate on the test set."""
        try:
            X_train, X_test, y_train, y_test = self.load_and_split_data()
            self.build_pipeline()
            self.pipeline.fit(X_train, y_train)
            logger.info("Model training completed.")
            y_pred = self.pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Classification Accuracy: {accuracy:.4f}")
            logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        except Exception as e:
            logger.exception(f"Error during training or evaluation: {e}")
            raise

if __name__ == '__main__':
    classifier = Classification()
    classifier.train_and_evaluate()
    new_query = ["What are the side effects of the COVID-19 vaccine?"]
    predicted_domain = classifier.pipeline.predict(new_query)[0]
    print(f"Predicted domain: {predicted_domain}")
