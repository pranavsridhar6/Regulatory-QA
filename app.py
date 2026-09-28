# app.py: web page for asking questions about FDA guidance or your own uploaded PDFs

import json
import os
import uuid

import streamlit as st

from main import (
    setup_qa_system, get_citations, docs_fingerprint,
    QA_PROMPT, UPLOAD_PROMPT, NO_ANSWER, UPLOAD_NO_ANSWER, INDEX_DIR,
)

# Where the FDA library lives, and where each session's uploads are saved
FDA_DOCS = r"C:\Projects\regulatory-qa\docs"
UPLOADS_ROOT = "uploads"

# --- Page settings: must be the first Streamlit command ---
st.set_page_config(page_title="Regulatory QA", page_icon="📄")
st.title("Regulatory QA")


# --- Build a QA chain once per document set and reuse it on every rerun ---
# fingerprint_key isn't used inside: it's there so Streamlit builds a new chain when the PDFs change
@st.cache_resource(max_entries=5)
def load_chain(mode, folder, index_dir, fingerprint_key):
    prompt = QA_PROMPT if mode == "FDA library" else UPLOAD_PROMPT
    return setup_qa_system(folder, index_dir=index_dir, qa_prompt=prompt)


# --- Show citations in a collapsible "Sources" box under an answer ---
def show_citations(citations):
    if citations:
        if any("[DRAFT]" in citation for citation in citations):
            st.warning("Some sources are drafts, not final versions. Check Sources below.")
        with st.expander(f"Sources ({len(citations)})"):
            for citation in citations:
                st.markdown(f"- {citation}")



# --- Sidebar: choose which documents to ask about ---
mode = st.sidebar.radio("Documents", ["FDA library", "My documents"])

if mode == "FDA library":
    folder = FDA_DOCS
    index_dir = INDEX_DIR
    no_answer = NO_ANSWER
    st.caption("Answers come only from the loaded FDA guidance documents, with citations.")
else:
    # Each browser session gets its own private upload folder
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    folder = os.path.join(UPLOADS_ROOT, st.session_state.session_id)
    index_dir = os.path.join(folder, "_index")
    no_answer = UPLOAD_NO_ANSWER
    os.makedirs(folder, exist_ok=True)

    uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    # Make the folder match the uploader: delete PDFs the user removed, save new ones
    keep = {f.name for f in uploaded_files}
    for name in os.listdir(folder):
        if name.lower().endswith(".pdf") and name not in keep:
            os.remove(os.path.join(folder, name))
    for f in uploaded_files:
        path = os.path.join(folder, f.name)
        if not os.path.exists(path):
            with open(path, "wb") as out:
                out.write(f.getbuffer())

    if not uploaded_files:
        st.info("Upload one or more PDFs in the sidebar to start.")
        st.stop()

    st.caption(f"Answers come only from your {len(uploaded_files)} uploaded document(s), with citations.")



# --- Load the chain for this document set (rebuilds only when the PDFs change) ---
fingerprint_key = json.dumps(docs_fingerprint(folder))
with st.spinner("Reading documents..."):
    qa_chain = load_chain(mode, folder, index_dir, fingerprint_key)

# --- Separate chat history for each mode, so switching doesn't mix conversations ---
history_key = f"messages_{mode}"
if history_key not in st.session_state:
    st.session_state[history_key] = []
messages = st.session_state[history_key]

# --- Redraw every earlier message on each rerun ---
for message in messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        show_citations(message.get("citations", []))

# --- New question ---
question = st.chat_input("Ask about the selected documents")
if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            answer = qa_chain.invoke(question)
        st.write(answer["result"])

        # No sources on a refusal
        citations = []
        if answer["result"].strip() != no_answer:
            citations = get_citations(answer)
        show_citations(citations)

    messages.append({"role": "assistant", "content": answer["result"], "citations": citations})