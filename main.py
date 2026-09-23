import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()


def setup_qa_system(folder_path):
    loader = PyPDFDirectoryLoader(folder_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever()
    llm = ChatAnthropic(model="claude-sonnet-4-6")

    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

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





