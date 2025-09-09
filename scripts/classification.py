import logging
from src.query_classification import Classification

if __name__ == "__main__":
    classifier = Classification()
    X_train, X_test, y_train, y_test = classifier.load_and_split_data()
    pipeline=classifier.build_pipeline()

    pipeline.fit(X_train, y_train)
    query = "How can I check my bank balance?"

    predicted_domain = pipeline.predict([query])[0]
    print(predicted_domain)
