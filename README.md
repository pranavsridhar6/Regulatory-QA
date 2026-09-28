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

## Draft 3

- **Automatic index rebuilds**: the index stores a fingerprint (name, size, modified time) of every PDF it was built from, and rebuilds itself when files are added, removed, or edited.
- **Hybrid search**: keyword search (BM25, with a tokenizer that keeps section codes like "1.3.3" intact) merged with meaning-based search.

| Retrieval setting | Chunks sent | Score |
|---|---|---|
| Meaning search, top 4 | 4 | 13/20 |
| Meaning search, top 8 | 8 | 16/20 |
| Hybrid, 4 keyword + 4 meaning | ≤ 8 | 15/20 |
| Hybrid, 4 keyword + 8 meaning | ≤ 12 | **17/20** |

At an equal budget of 8 chunks, hybrid search lost two questions whose answers ranked 5th–8th in meaning search. Adding keyword results on top of the full meaning results gained one question with no regressions, at the cost of up to 12 chunks per answer. With 20 questions, a one-question difference is within noise.

**Remaining failures:** q09 and q15 fail because the page-by-page splitter separates related bullets across a page break (the answers sit at the bottom of one page; the retrieved chunk starts the next). q08 retrieves the right passage but without its section label.

## Draft 4

- **Automatic draft/final detection**: each PDF's status is read from its first 3 pages ("Not for Implementation", "distributed for comment purposes only"). A plain search for "draft" would mislabel final guidances that cite other drafts. PDFs that aren't FDA guidance are tagged N/A.
- **Line-number cleanup**: two PDFs had printed margin line numbers mixed into the text. The cleaner finds each page's longest unbroken run of numbers (20+) and removes only numbers inside that run, so table values, footnote markers, and pages without line numbers are left untouched. Tested on all 9 PDFs: 0 pages changed in the 7 without line numbers.
- **Corpus expanded to 9 documents (263 pages)**, including a 2017 **draft** Q&A that overlaps the final RTR guidance.
- **Draft-aware answers**: each excerpt reaches the model labeled [FINAL] or [DRAFT], with instructions to prefer final guidance and explicitly flag anything drawn from a draft.

| Setting | Original 20 | Draft check (q21) |
|---|---|---|
| 6 docs, hybrid retrieval | 17/20 | n/a |
| 9 docs, status hidden from model | 15/20 | n/a |
| 9 docs, status shown to model | 16/20 | PASS |

Before the fix, the system stated a draft-only rule ("two API lots for three batches") as a binding requirement. After, it labels it as draft guidance. Adding the draft Q&A cost two questions (q02, q07): its chunks cover the same topics as the final RTR guidance and crowd final-guidance chunks out of the retrieval slots. When final text isn't retrieved, answers can still lead with the draft point before the caveat.

## Draft 5

- **Streamlit chat UI** (`app.py`): run with `python -m streamlit run app.py`.
- The QA chain is built once per server with `@st.cache_resource`; chat history is kept in `st.session_state` so it survives Streamlit's rerun-on-every-interaction model.
- Each answer shows a collapsible Sources list (title, draft/final status, PDF page), no sources on refusals, and a warning banner when any source is draft guidance.
- Citation formatting lives in one function (`get_citations` in `main.py`) shared by the terminal chat and the web UI.
- Streamlit's file watcher is disabled in `.streamlit/config.toml`: it scanned every module in `transformers` and logged hundreds of harmless `torchvision` import errors. Restart the app after code changes.