import os
import logging
from src.vectorstore_builder import VectorStoreBuilder


def main():
    builder = VectorStoreBuilder()
    domain_vectorstores = builder.build_and_save()

    sample_domain = list(domain_vectorstores.keys())[0]

    vs = builder.load(sample_domain)

    retriever = vs.as_retriever(search_kwargs={"k": 2})
    query = "What are the side effects of the COVID-19 vaccine?"
    results = retriever.get_relevant_documents(query)

    print(f"Query: {query}")
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Content:\n{doc.page_content}")
        print(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    main()
