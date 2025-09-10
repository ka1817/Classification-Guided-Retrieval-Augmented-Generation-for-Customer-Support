import logging
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.query_classification import Classification
from src.vectorstore_builder import VectorStoreBuilder
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("QueryRouter")

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


class QueryRouter:
    def __init__(self):
        self.classifier = Classification()
        try:
            self.classifier.load_pipeline()
            logger.info("✅ Classifier loaded successfully.")
        except FileNotFoundError:
            logger.warning("No classifier pipeline found. Training a new classifier...")
            self.classifier.train_and_evaluate()
            logger.info("✅ Classifier trained and saved successfully.")

        self.vs_builder = VectorStoreBuilder()

        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", max_tokens=150)
        logger.info("✅ LLM loaded successfully.")

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a helpful assistant.

Use the context below to answer the question. Only use the information from the context. 
If the context does not provide enough info, say you don’t know.

Context:
{context}

Question:
{question}

Answer clearly and concisely:
            """,
        )

    def route(self, query: str) -> tuple[str, str]:
        predicted_domain = self.classifier.pipeline.predict([query])[0]
        logger.info(f"Predicted domain: {predicted_domain}")

        try:
            vs = self.vs_builder.load(predicted_domain)
            logger.info(f"✅ Loaded {predicted_domain} VectorStore")
        except FileNotFoundError:
            logger.warning(f"No vectorstore found for domain '{predicted_domain}'. Building now...")
            domain_vectorstores = self.vs_builder.build_and_save()
            if predicted_domain in domain_vectorstores:
                vs = domain_vectorstores[predicted_domain]
                logger.info(f"✅ VectorStore built and loaded for domain '{predicted_domain}'")
            else:
                logger.error(f"Failed to build vectorstore for domain '{predicted_domain}'")
                return predicted_domain, "No vectorstore available for this domain."

        retriever = vs.as_retriever(search_kwargs={"k": 3})

        rag_chain = (
            RunnableParallel({
                "context": retriever | RunnableLambda(
                    lambda chunks: "\n\n".join([d.page_content for d in chunks])
                ),
                "question": RunnablePassthrough()
            })
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke(query)
        return predicted_domain, answer


