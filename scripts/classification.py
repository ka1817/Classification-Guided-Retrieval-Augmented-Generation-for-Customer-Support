import logging
from src.query_classification import Classification

if __name__ == "__main__":
    classifier = Classification()

    classifier.train_and_evaluate()

    new_query = ["Where should i go to ask about bank details"]
    predicted_domain = classifier.pipeline.predict(new_query)[0]
    print(f"Predicted domain: {predicted_domain}")

