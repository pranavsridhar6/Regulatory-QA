# eval.py: runs every question in eval_questions.json through the QA chain and scores the answers
import json
import os

from main import setup_qa_system, NO_ANSWER

# --- Load the eval questions from the JSON file ---
with open("eval_questions.json", encoding="utf-8") as f:
    questions = json.load(f)

print(f"Loaded {len(questions)} questions")

# --- Score one result. Returns (True/False, reason it failed) ---
def grade(result):
    q = result["q"]
    answer = result["answer"]

    # Out-of-corpus: passes only if the answer is exactly the refusal sentence
    if q["type"] == "out_of_corpus":
        if answer == NO_ANSWER:
            return True, ""
        return False, "did not refuse"

    # In-corpus and tricky: every key-fact group needs at least one matching phrase
    answer_lower = answer.lower()
    for group in q["key_facts"]:
        if not any(phrase.lower() in answer_lower for phrase in group):
            return False, f"missing: {group[0]}"

    # The expected document must be among the retrieved sources
    if q["expected_doc"] not in result["cited_files"]:
        return False, "expected doc not retrieved"

    return True, ""


# --- Build the same QA chain main.py uses (loads the saved index, uses the answer cache) ---
qa_chain = setup_qa_system(r"C:\Projects\regulatory-qa\docs")

# --- Ask every question and keep the answer plus the files it cited ---
results = []
for q in questions:
    answer = qa_chain.invoke(q["question"])
    answer_text = answer["result"].strip()

    # File names of the chunks retrieved for this question, no duplicates
    cited_files = {os.path.basename(doc.metadata["source"]) for doc in answer["source_documents"]}

    results.append({"q": q, "answer": answer_text, "cited_files": cited_files})
    print(f"{q['id']} done")

# --- Score every result and print one line per question ---
passed_count = 0
for r in results:
    passed, reason = grade(r)
    r["passed"] = passed
    r["reason"] = reason
    if passed:
        passed_count += 1

    status = "PASS" if passed else "FAIL"
    flag = "  <- tricky: read this answer yourself" if r["q"]["type"] == "tricky" else ""
    print(f"{r['q']['id']} {status} {reason}{flag}")

    # Show the full answer for failures and tricky questions so you can judge them
    if not passed or r["q"]["type"] == "tricky":
        print("    " + r["answer"])
        print("    Retrieved from:", sorted(r["cited_files"]))

print(f"\nScore: {passed_count}/{len(results)}")

# --- Write the results as a Markdown table, ready to paste into the README ---
lines = [
    f"**Score: {passed_count}/{len(results)}**",
    "",
    "| ID | Type | Question | Result | Note |",
    "|---|---|---|---|---|",
]
for r in results:
    q = r["q"]
    status = "PASS" if r["passed"] else "FAIL"
    lines.append(f"| {q['id']} | {q['type']} | {q['question']} | {status} | {r['reason']} |")

with open("eval_results.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("Wrote eval_results.md")