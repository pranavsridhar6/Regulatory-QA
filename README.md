## Draft 2

### What changed
- **Citations on every answer**: document title, draft/final status, and PDF page number. Sources are hidden when the system declines to answer.
- **Draft vs final tagging**: each document is tagged in a lookup table; untagged files show as `[UNTAGGED]` rather than passing as final. All 6 current documents are final.
- **Custom prompt**: answers start directly, use plain text, and use one fixed sentence when the documents don't cover a question.
- **Loader fix**: the PDF loader was reading `docs/later/` (parked files) because its default pattern searches subfolders. Citations exposed this. Corpus is now 6 documents, 177 pages.
- **Repeatable output**: the model rejects the `temperature` setting, so repeatability comes from a saved FAISS index plus a SQLite cache of model answers. Re-running the eval replays identical answers.

### Evaluation
20 questions with known answers, each tied to a PDF page that I checked by hand: 14 in-corpus, 3 "tricky" (easily confused numbers), 3 out-of-corpus. A question passes if the answer contains every required key fact (alternate wordings allowed) and the expected document was retrieved. Out-of-corpus questions pass only on the exact refusal sentence.

| Retrieval setting | Score |
|---|---|
| Top 4 chunks (baseline) | 13/20 |
| Top 8 chunks | 16/20 |

Top 8 fixed 3 questions with no regressions. Full tables: [`eval_results_baseline.md`](eval_results_baseline.md), [`eval_results_k8.md`](eval_results_k8.md).

The 4 remaining failures all retrieved the correct document but not the passage containing the answer (e.g., a footnote, or a section code like "1.3.3"). All 4 declined rather than answering incorrectly.

### Known limitations
- Key-fact grading checks that the right facts appear, not that they are matched to the right case. A swapped answer ("24 in general, 12 for highly variable") would pass, so the 3 tricky answers are reviewed by hand.
- The source check is per document, not per page.
- Two key facts (q04, q05) were widened after manual review: the answers were correct but used "sameness" and "prohibit" rather than the original key words.
- Citations list the chunks retrieved, not only the ones the model used.
- Page numbers are PDF page positions, not printed page numbers. The two differ by several pages because of unnumbered title and contents pages.
- Top-8 was chosen using these same 20 questions, so the score may be slightly optimistic for new questions.

### Next
Keyword + semantic (hybrid) search for exact codes, reranking, cleanup of the 3 parked PDFs (including tagging the 2017 draft), then LangGraph and a Streamlit UI.