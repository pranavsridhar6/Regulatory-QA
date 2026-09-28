import os
import warnings


from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
import json
import re
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


load_dotenv()

# --- Hide LangChain's harmless "allowed_objects" notice that prints on every cache read ---
warnings.filterwarnings("ignore", message=".*allowed_objects")

DOC_INFO = {
    '54373781fnl_Controlled Correspondence Related to Generic Drug Development.pdf':
        {'title': 'Controlled Correspondence Related to Generic Drug Development', 'status': 'FINAL'},
    '54374223fnl_bioequivalence_studies_with_pharmacokinetic_endpoints_for_drugs_submitted_under_an_anda.pdf':
        {'title': 'Bioequivalence Studies With PK Endpoints for Drugs Submitted Under an ANDA', 'status': 'FINAL'},
    'ANDA-Submissions----Refuse-to-Receive-Standards-Rev.2.pdf':
        {'title': 'ANDA Submissions: Refuse-to-Receive Standards (Rev. 2)', 'status': 'FINAL'},
    'GUI_FINAL_GoodANDASubmissionPractices_Published_Jan 2022.pdf':
        {'title': 'Good ANDA Submission Practices (Jan 2022)', 'status': 'FINAL'},
    'GUI_Final_level 2_ ANDA Submissions - Content and Format_Revised_June_2019_0.pdf':
        {'title': 'ANDA Submissions: Content and Format (Rev. 1, June 2019)', 'status': 'FINAL'},
    'GUI_Final_Referencing_Approved _Oct 2020.pdf':
        {'title': 'Referencing Approved Drug Products in ANDA Submissions (Oct 2020)', 'status': 'FINAL'},
    '58381904fnl_Statistical Approaches to Establishing Bioequivalence.pdf':
        {'title': 'Statistical Approaches to Establishing Bioequivalence (May 2026)'},
    'ANDA-Submissions---Refuse-to-Receive-Standards----Questions-and-Answers-Guidance-for-Industry.pdf':
        {'title': 'ANDA Submissions: Refuse-to-Receive Standards Q&A (Oct 2017)'},
    'Submission-of-Summary-Bioequivalence-Data-for-Abbreviated-New-Drug-Applications.pdf':
        {'title': 'Submission of Summary Bioequivalence Data for ANDAs (May 2011)'},
}

NO_ANSWER = "The loaded FDA guidance documents do not address this."

PROMPT_TEXT = """You answer questions about FDA generic-drug (ANDA) guidance documents.

Use only the excerpts below. Start directly with the answer. Do not open with phrases like "Based on the provided context", and do not mention the excerpts or the context.

Each excerpt is labeled [FINAL] or [DRAFT]. Draft guidance is not for implementation and is not FDA's current policy. Base your answer on FINAL excerpts. 

Use a DRAFT excerpt only if no FINAL excerpt covers the point, and then state clearly that it comes from draft guidance that is not final. 

If a DRAFT excerpt conflicts with a FINAL excerpt, follow the FINAL one.

If the excerpts answer only part of the question, answer that part and say which part the documents do not cover.

If the excerpts do not answer the question at all, reply with exactly: "The loaded FDA guidance documents do not address this."

Write in plain text. No markdown headers and no bold.

Excerpts:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(template=PROMPT_TEXT, input_variables=["context", "question"])

# How each retrieved chunk appears inside {context}: its status label, then its text
DOC_PROMPT = PromptTemplate(template="[{status}]\n{page_content}", input_variables=["page_content", "status"])

# Folder where the search index is saved, so we don't rebuild it every run
INDEX_DIR = "faiss_index"
# File inside the index folder recording which PDFs the index was built from
FINGERPRINT_FILE = os.path.join(INDEX_DIR, "fingerprint.json")

# File where Claude's answers are saved. Same prompt in = saved answer out, no new API call.
# Delete this file to force fresh answers from Claude.
CACHE_PATH = "llm_cache.db"

# How many chunks the search returns per question. Default was 4. Change this to experiment.
TOP_K = 8

# Bump this number whenever the index-building code changes, so old indexes rebuild automatically
INDEX_VERSION = 3

# Phrases FDA prints on draft guidance. Plain "draft" is not enough: final guidances often cite other drafts.
DRAFT_MARKERS = ["not for implementation", "distributed for comment purposes only"]

# Phrases found on FDA guidance generally (draft or final)
GUIDANCE_MARKERS = ["contains nonbinding recommendations", "guidance for industry"]


# --- Answer cache: makes repeat questions return identical answers (needed for a repeatable eval) ---
set_llm_cache(SQLiteCache(database_path=CACHE_PATH))


# --- Split text into lowercase words for keyword search. Keeps section codes like "1.3.3" whole. ---
def simple_tokenize(text):
    return re.findall(r"\w+(?:\.\w+)*", text.lower())

# --- Line-number cleanup for PDFs with printed margin line numbers ---
# A number at the end of a line (or alone on a line), and a number at the start of a line
TRAILING_NUMBER = re.compile(r"(?:^|\s)(\d{1,4})\s*$")
LEADING_NUMBER = re.compile(r"^\s*(\d{1,4})\s+(?=\S)")


# Find the longest stretch of consecutive whole numbers in a list, e.g. 32, 33, ..., 76
def longest_run(numbers):
    present = set(numbers)
    best = (0, 0)
    for n in present:
        if n - 1 not in present:
            high = n
            while high + 1 in present:
                high += 1
            if high - n > best[1] - best[0]:
                best = (n, high)
    return best


# Remove printed line numbers from one page of text. Pages without them come back unchanged.
def clean_line_numbers(text):
    lines = text.split("\n")

    # Collect every number sitting at the start or end of a line
    candidates = []
    for line in lines:
        for pattern in (TRAILING_NUMBER, LEADING_NUMBER):
            match = pattern.search(line)
            if match:
                candidates.append(int(match.group(1)))

    # Line numbers form a long unbroken run. No run of 20+ means no line numbers: leave the page alone.
    low, high = longest_run(candidates)
    if high - low + 1 < 20:
        return text

    # Remove only numbers inside that run, so table values and footnote markers survive
    cleaned = []
    for line in lines:
        match = TRAILING_NUMBER.search(line)
        if match and low <= int(match.group(1)) <= high:
            line = line[:match.start()]
        match = LEADING_NUMBER.search(line)
        if match and low <= int(match.group(1)) <= high:
            line = line[match.end():]
        cleaned.append(line)
    return "\n".join(cleaned)


# --- Work out a document's status from the text of its first 3 pages ---
def detect_status(pages):
    text = " ".join(page.page_content for page in pages[:3])
    text = " ".join(text.split()).lower()
    if any(marker in text for marker in DRAFT_MARKERS):
        return "DRAFT"
    if any(marker in text for marker in GUIDANCE_MARKERS):
        return "FINAL"
    return "N/A"

# --- Fingerprint of the docs folder: each PDF's name, size, and last-modified time ---
# If any PDF is added, removed, or edited, the fingerprint changes and the index rebuilds.
def docs_fingerprint(folder_path):
    files = []
    for name in sorted(os.listdir(folder_path)):
        if name.lower().endswith(".pdf"):
            stats = os.stat(os.path.join(folder_path, name))
            files.append([name, stats.st_size, int(stats.st_mtime)])
    return {"version": INDEX_VERSION, "files": files}




def setup_qa_system(folder_path):
# Embedder: turns text into 384 numbers that capture its meaning
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # --- Compare the PDFs in the folder now against the ones the saved index was built from ---
    current = docs_fingerprint(folder_path)
    saved = None
    if os.path.exists(FINGERPRINT_FILE):
        with open(FINGERPRINT_FILE, encoding="utf-8") as f:
            saved = json.load(f)

    if saved == current:
        # A saved index exists: load it instead of re-reading all the PDFs
        vector_store = FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
        print("Loaded saved index")
    else:
        # No saved index, or the PDFs/version changed: read the PDFs in docs\ only (not docs\later\)
        loader = PyPDFDirectoryLoader(folder_path, glob="*.pdf")
        documents = loader.load()
        print(f"Loaded {len(documents)} pages")

        # --- Strip printed margin line numbers (pages without them are left unchanged) ---
        for page in documents:
            page.page_content = clean_line_numbers(page.page_content)

        # --- Group pages by file, detect each file's status, and label every page with it ---
        pages_by_file = {}
        for page in documents:
            pages_by_file.setdefault(page.metadata["source"], []).append(page)
        for source, pages in pages_by_file.items():
            status = detect_status(pages)
            for page in pages:
                page.metadata["status"] = status

        # Cut pages into 1000-character chunks that overlap by 200 characters
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Turn every chunk into numbers, build the index, and save it to disk
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(INDEX_DIR)

        # Record which PDFs this index was built from, for the check on the next run
        with open(FINGERPRINT_FILE, "w", encoding="utf-8") as f:
            json.dump(current, f)
        print("Built and saved new index")


    # --- Keyword search (BM25): scores chunks by shared words, rare words count most ---
    all_chunks = list(vector_store.docstore._dict.values())
    keyword_retriever = BM25Retriever.from_documents(all_chunks, preprocess_func=simple_tokenize)
    keyword_retriever.k = TOP_K // 2

    # --- Meaning search (FAISS): finds chunks with similar meaning ---
    meaning_retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    # --- Hybrid: run both searches and merge their results into one ranked list ---
    retriever = EnsembleRetriever(retrievers=[keyword_retriever, meaning_retriever], weights=[0.5, 0.5])
    llm = ChatAnthropic(model="claude-sonnet-4-6")

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT, "document_prompt": DOC_PROMPT},
    )

    return qa_chain

if __name__ == '__main__':
    qa_chain = setup_qa_system(r"C:\Projects\regulatory-qa\docs")

    while True:
        question = input('\nAsk a question: ')
        if question.lower() == 'exit':
            break

        answer = qa_chain.invoke(question)

        print('Answer:')
        print(answer['result'])
        if answer['result'].strip() == NO_ANSWER:
            continue
        print('\nSource Documents:')
        seen=set()
        for doc in answer['source_documents']:
            file_name = os.path.basename(doc.metadata['source'])
            page_number = doc.metadata['page'] + 1
            # Title from DOC_INFO if listed, otherwise the file name. Status comes from the PDF text.
            title = DOC_INFO.get(file_name, {}).get('title', file_name)
            status = doc.metadata.get('status', 'UNTAGGED')
            citation = f"{title} [{status}], PDF Page: {page_number}"
            if citation not in seen:
                seen.add(citation)
                print(citation)





