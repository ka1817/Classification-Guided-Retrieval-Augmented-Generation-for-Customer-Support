from src.query_classification import Classification
if __name__ == "__main__":
    classifier = Classification()
    classifier.load_pipeline()   

    new_query = "How can I check my account balance?"
    predicted_domain = classifier.pipeline.predict([new_query])[0]
    print(f"Predicted domain: {predicted_domain}")
