# app.py: web page for asking questions about the loaded FDA guidance documents

import streamlit as st

from main import setup_qa_system

# --- Page settings: browser tab title and icon. Must be the first Streamlit command. ---
st.set_page_config(page_title="Regulatory QA", page_icon="📄")

st.title("Regulatory QA")
st.caption("Answers come only from the loaded FDA guidance documents, with citations.")


# --- Build the QA chain once, then reuse it on every rerun ---
@st.cache_resource
def load_chain():
    return setup_qa_system(r"C:\Projects\regulatory-qa\docs")


qa_chain = load_chain()

# --- Chat history: stored in session_state so it survives reruns ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Redraw every earlier message on each rerun ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

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
    st.session_state.messages.append({"role": "assistant", "content": answer["result"]})