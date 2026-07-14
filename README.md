# intelligent-document-qa

A production-style RAG pipeline that lets you query any collection of documents using natural language. Built with ChromaDB for vector storage, FastAPI for the query layer and Claude (Anthropic) as the reasoning backbone. Comes with a lightweight Streamlit dashboard to evaluate retrieval quality.

---

## Why I built this

Most RAG tutorials show you how to ask questions about a single PDF. That's fine for a demo but it doesn't tell you anything about whether your chunking strategy is actually working or whether your retrieval is pulling the right context before the LLM even sees it.

This project treats evaluation as a first class concern. You can swap chunking strategies, compare retrieval precision across runs and see latency per query, all from a simple UI. It's the kind of thing you'd actually want before putting a system like this in front of real users.

---

## What's inside

```
intelligent-document-qa/
├── src/
│   ├── ingestion/          # document loading, chunking, embedding
│   ├── retrieval/          # query engine, reranking, context assembly
│   └── evaluation/         # precision metrics, faithfulness scoring
├── api/                    # FastAPI app — query endpoints + health check
├── ui/                     # Streamlit dashboard
├── tests/                  # pytest suite
├── data/sample_docs/       # drop your PDFs/TXTs here
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

**1. Clone and install**

```bash
git clone https://github.com/Saravanan-vasudevan/intelligent-document-qa
cd intelligent-document-qa
pip install -r requirements.txt
```

**2. Set your API key**

```bash
cp .env.example .env
```

**3. Ingest some documents**

Drop PDFs or text files into `data/sample_docs/`, then:

```bash
python -m src.ingestion.pipeline --source data/sample_docs/
```

**4. Start the API**

```bash
uvicorn api.main:app --reload --port 8000
```

**5. Launch the dashboard**

```bash
streamlit run ui/dashboard.py
```

Or run everything at once with Docker:

```bash
docker-compose up --build
```

---

## Tech stack

| Layer | Tool |
|---|---|
| LLM | Anthropic Claude (claude-sonnet-4-6) |
| Vector store | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Containerisation | Docker + Docker Compose |
| Testing | pytest |

---

## API reference

Once the server is running, hit `http://localhost:8000/docs` for the interactive Swagger UI.

**POST `/query`**
```json
{
  "question": "What are the key findings in the report?",
  "top_k": 5,
  "strategy": "semantic"
}
```

**GET `/health`** — basic health check

**GET `/collections`** — list all document collections currently indexed

---

## Evaluation metrics

The dashboard tracks three things per query run:

- **Retrieval precision** — are the top-k chunks actually relevant to the question?
- **Answer faithfulness** — does the LLM's answer stay grounded in the retrieved context, or is it hallucinating?
- **Latency** — end-to-end time from query to response

You can export any evaluation run to CSV for offline analysis.

---

## Chunking strategies

Three strategies are available, configurable per ingestion run:

- `fixed` — fixed token size with overlap (fast, good baseline)
- `semantic` — splits on sentence boundaries, groups by semantic similarity
- `sliding` — overlapping windows, useful for dense technical documents

The evaluation dashboard lets you compare retrieval precision across strategies on the same query set.

---

## Notes

- ChromaDB stores everything locally under `data/chroma_store/` by default. No external database needed to get started.
- The project uses `all-MiniLM-L6-v2` for embeddings because it's fast and runs comfortably on CPU. If you have a GPU available, swapping to a larger model is a one-line change in `src/ingestion/embedder.py`.
- Evaluation scoring uses a secondary Claude call to judge faithfulness — this means eval runs cost a small number of API tokens. Keep that in mind if you're on a limited plan.

---
