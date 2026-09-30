# PDF RAG Assistant

Upload your PDFs and ask questions about them. Answers come only from the documents you select, each with page-level citations, and the assistant declines when the documents don't contain the answer. Try it on the built-in library of FDA guidance documents, or upload your own.

> **For regulatory teams:** built around FDA guidance. It detects draft guidance automatically and flags it in every answer that relies on it, so draft recommendations are never presented as final policy. Every claim cites the document and page, so you can verify it in seconds.

## Features

- Hybrid keyword and semantic search, so exact terms (section numbers, defined terms) are found as well as paraphrases
- Upload and search your own PDFs, with one-click document summaries
- Page-level citations, with draft sources clearly labeled
- Declines questions the documents don't answer, even when the model knows the answer from elsewhere

## Run

Install dependencies with `pip install -r requirements.txt`, create a `.env` file containing `ANTHROPIC_API_KEY=your-key`, then start the app:

    python -m streamlit run app.py

Open http://localhost:8501.

## Evaluation

On a 20-question FDA guidance benchmark with hand-verified answers: 16/20 correct, 3/3 out-of-corpus questions correctly declined, and draft-only guidance correctly flagged as draft. See [evaluation results](eval_results_draft_aware.md).

**Stack:** Python, Streamlit, LangChain, FAISS, BM25, local Hugging Face embeddings, and Claude.