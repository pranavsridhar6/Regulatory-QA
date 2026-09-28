# app.py: web page for asking questions about the loaded FDA guidance documents

import streamlit as st

from main import setup_qa_system, get_citations, NO_ANSWER

# --- Page settings: browser tab title and icon. Must be the first Streamlit command. ---
st.set_page_config(page_title="Regulatory QA", page_icon="📄")

st.title("Regulatory QA")
st.caption("Answers come only from the loaded FDA guidance documents, with citations.")


# --- Build the QA chain once, then reuse it on every rerun ---
@st.cache_resource
def load_chain():
    return setup_qa_system(r"C:\Projects\regulatory-qa\docs")


qa_chain = load_chain()


# --- Show citations in a collapsible "Sources" box under an answer ---
def show_citations(citations):
    if citations:
        # Warn visibly when any source is draft guidance
        if any("[DRAFT]" in citation for citation in citations):
            st.warning("Some sources are draft guidance, which is not final FDA policy. Check Sources below.")
        with st.expander(f"Sources ({len(citations)})"):
            for citation in citations:
                st.markdown(f"- {citation}")


# --- Chat history: stored in session_state so it survives reruns ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Redraw every earlier message on each rerun ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        show_citations(message.get("citations", []))

# --- New question: a chat box pinned to the bottom of the page ---
question = st.chat_input("Ask about the loaded FDA guidance")
if question:
    # Show the question, and remember it
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Get the answer, show it, and remember it
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            answer = qa_chain.invoke(question)
            st.write(answer["result"])

        # No sources on a refusal, same as the terminal chat
        citations = []
        if answer["result"].strip() != NO_ANSWER:
            citations = get_citations(answer)
        show_citations(citations)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer["result"], "citations": citations}
    )