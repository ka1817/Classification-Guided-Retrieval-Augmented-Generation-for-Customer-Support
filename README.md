# 🚀 QueryRouter-RAG

## 🔥 Introduction

In customer support and knowledge-driven applications, users often ask queries that belong to multiple business domains (e.g., insurance, banking, retail, healthcare). Traditional chatbots either rely on **static FAQs** or a **single-domain knowledge base**, which makes them brittle, inaccurate, and frustrating for users.

To solve this, we built **QueryRouter-RAG**, a domain-aware Retrieval-Augmented Generation (RAG) system that:

1. **Classifies** a user query into its correct business domain.
2. **Retrieves** the most relevant documents from a **domain-specific FAISS vectorstore**.
3. **Generates** accurate, context-aware answers using **Groq-powered LLMs**.

This ensures that every user gets the **right answer from the right knowledge base**, improving both accuracy and user trust.

---

## 🎯 Problem We Solved

* ❌ **One-size-fits-all chatbots** → Often fail when queries span multiple domains.
* ❌ **Hard-coded rules** → Require constant manual updates and don’t scale.
* ❌ **Knowledge silos** → A single vector database mixes all domains, leading to irrelevant results.

✅ With **QueryRouter-RAG**, we solve these challenges:

* ✅ **Domain Classification** → Automatically predicts which domain (e.g., banking, insurance, retail) a query belongs to.
* ✅ **Domain-specific Retrieval** → Builds separate FAISS vectorstores for each domain, ensuring relevant and precise results.
* ✅ **Scalable & Modular** → Easy to extend with new domains, retrievers, or models.
* ✅ **Production-ready** → FastAPI backend, Dockerized, and CI/CD integrated.

---

## ✨ Features

* 🔎 **Query Classification** → Classifies user queries into pre-defined domains using **Naive Bayes** with TF-IDF.
* 📚 **Domain-wise Vector Stores** → Builds **FAISS** vector stores for each domain.
* 🤖 **RAG-powered Generation** → Retrieves relevant knowledge chunks and generates responses with **Groq LLMs**.
* ⚡ **FastAPI Backend** → Exposes APIs for query routing and RAG-based response generation.
* 🎨 **Frontend UI** → Simple web interface built with **Jinja2 + HTML/CSS**.
* 🐳 **Containerized** → Ready-to-deploy with Docker.
* 🔄 **CI/CD** → Automated testing and Docker image publishing via **GitHub Actions**.

---

## 🏗️ Project Structure

```
.
├── .github/workflows/            # CI/CD pipelines
├── data/                         # Dataset (CSV with queries, responses, intents, domains)
├── models/                       # Saved ML models (classifier pipeline, etc.)
├── scripts/                      # Utility or training scripts
├── src/                          # Core source code
│   ├── data_ingestion.py         # Load dataset
│   ├── data_preprocessing.py     # Splitting & chunking into LangChain Document objects
│   ├── query_classification.py   # ML classifier for domain prediction
│   ├── vectorstore_builder.py    # Build & save FAISS vectorstores per domain
│   ├── generation.py             # RAG pipeline (retriever + Groq LLM)
│   └── __init__.py
│
├── static/                       # Static files (CSS, JS)
│   └── style.css
├── templates/                    # Web templates
│   └── index.html
├── tests/                        # Unit tests for CI
├── vectorstores/                 # Saved FAISS vectorstores per domain
│
├── .dockerignore                 # Docker ignore file
├── .gitignore                    # Git ignore file
├── Dockerfile                    # Containerization
├── README.md                     # Project documentation
├── main.py                       # FastAPI entrypoint
└── requirements.txt              # Python dependencies
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/ka1817/queryrouter-rag.git
cd queryrouter-rag
```

### 2️⃣ Setup Python Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
```

---

## ▶️ Running the App

### Local Development

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

App will be available at 👉 **[http://localhost:8000](http://localhost:8000)**

### Query Example (API)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"query": "What is my insurance claim status?"}'
```

Response:

```json
{
  "domain": "insurance",
  "answer": "Your claim is under processing..."
}
```

### Frontend UI

Navigate to 👉 **[http://localhost:8000/](http://localhost:8000/)**

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t queryrouter-rag .
```

### Run Container

```bash
docker run -p 8000:8000 queryrouter-rag
```

---

## 🔄 CI/CD with GitHub Actions

* ✅ Runs unit tests on every push/PR to `main`
* ✅ Builds and pushes Docker image to **Docker Hub** if tests pass
* Configured in **.github/workflows/cicd.yml**

Secrets required in GitHub repo settings:

* `GROQ_API_KEY`
* `LANGCHAIN_API_KEY`
* `DOCKER_USERNAME`
* `DOCKER_PASSWORD`

---

## 🧠 Tech Stack

* **Backend**: FastAPI
* **LLM**: Groq + LangChain
* **Vector DB**: FAISS
* **ML**: Scikit-learn (Naive Bayes, TF-IDF)
* **Orchestration**: RunnableChain (LangChain)
* **CI/CD**: GitHub Actions
* **Containerization**: Docker

---

## 👨‍💻 Author

Pranav Reddy [ka1817](https://github.com/ka1817)
