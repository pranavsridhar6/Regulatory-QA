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
}

NO_ANSWER = "The loaded FDA guidance documents do not address this."

PROMPT_TEXT = """You answer questions about FDA generic-drug (ANDA) guidance documents.

Use only the excerpts below. Start directly with the answer. Do not open with phrases like "Based on the provided context", and do not mention the excerpts or the context.

If the excerpts answer only part of the question, answer that part and say which part the documents do not cover.

If the excerpts do not answer the question at all, reply with exactly: "The loaded FDA guidance documents do not address this."

Write in plain text. No markdown headers and no bold.

Excerpts:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(template=PROMPT_TEXT, input_variables=["context", "question"])

# Folder where the search index is saved, so we don't rebuild it every run
INDEX_DIR = "faiss_index"

# File where Claude's answers are saved. Same prompt in = saved answer out, no new API call.
# Delete this file to force fresh answers from Claude.
CACHE_PATH = "llm_cache.db"
# --- Answer cache: makes repeat questions return identical answers (needed for a repeatable eval) ---
set_llm_cache(SQLiteCache(database_path=CACHE_PATH))

def setup_qa_system(folder_path):
    # Embedder: turns text into 384 numbers that capture its meaning
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(INDEX_DIR):
        # A saved index exists: load it instead of re-reading all the PDFs
        vector_store = FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
        print("Loaded saved index")
    else:
        # No saved index yet: read the PDFs in docs\ only (not docs\later\)
        loader = PyPDFDirectoryLoader(folder_path, glob="*.pdf")
        documents = loader.load()
        print(f"Loaded {len(documents)} pages")

        # Cut pages into 1000-character chunks that overlap by 200 characters
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Turn every chunk into numbers, build the index, and save it to disk
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(INDEX_DIR)
        print("Built and saved new index")

    retriever = vector_store.as_retriever()
    llm = ChatAnthropic(model="claude-sonnet-4-6")

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
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
            info = DOC_INFO.get(file_name, {'title': file_name, 'status': 'UNTAGGED'})
            citation = f"{info['title']} [{info['status']}], PDF Page: {page_number}"
            if citation not in seen:
                seen.add(citation)
                print(citation)





