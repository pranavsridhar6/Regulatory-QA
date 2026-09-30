# PDF Parser 

PDF parser that lets you upload your PDFs and ask any questions related to them you want. Answers are 100% grounded in the selected documents and include page citations; draft sources are clearly labeled. Feel free to play around with **default Regulatory FDA PDFs** as a sample or click upload documents and get started with analyzing your own documents.

## Features

- Hybrid keyword and semantic search across 9 FDA generic-drug guidance documents
- Upload and search your own PDFs, with document summaries
- Refuses questions the documents do not answer

## Run

Install dependencies with `pip install -r requirements.txt`, set your `ANTHROPIC_API_KEY`, then start the app:

```bash
python -m streamlit run app.py
```

Open http://localhost:8501.

## Evaluation

Scores **17/20** on the 20-question FDA guidance benchmark. See [evaluation results](eval_results_hybrid12.md).

**Stack:** Python, Streamlit, LangChain, FAISS, BM25, local Hugging Face embeddings, and Claude.
