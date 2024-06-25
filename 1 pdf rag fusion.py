import os
import utils
import streamlit as st
from streaming import StreamHandler
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from dotenv import load_dotenv
import tempfile
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain.agents import create_vectorstore_agent
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from typing import List, Dict, Any
from operator import itemgetter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()
st.set_page_config(page_title="ThinkAI", page_icon="📄")
st.subheader(
    "Load your PDF, ask questions, and receive answers directly from the document."
)

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def generate_rag_fusion_queries(question: str) -> List[str]:
    """Generates multiple search queries related to a given question using RAG-Fusion."""
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    generate_queries = (
        prompt_rag_fusion
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )
    return generate_queries.invoke({"question": question})


def reciprocal_rank_fusion(results: List[List], k: int = 60) -> List[tuple]:
    """Reciprocal rank fusion for multiple lists of ranked documents."""
    fused_scores: Dict[str, float] = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    return [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]


def run_rag_fusion(retriever, question: str, llm) -> str:
    """Runs the complete RAG-Fusion pipeline, including retrieval, fusion, and answer generation."""
    retrieval_chain_rag_fusion = (
        generate_rag_fusion_queries | retriever.map() | reciprocal_rank_fusion
    )
    docs = retrieval_chain_rag_fusion.invoke({"question": question})

    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        {"context": retrieval_chain_rag_fusion, "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )
    return final_rag_chain.invoke({"question": question})


# Load blog
def load_document(file_path):
    if file_path:
        loader = PyPDFLoader(file_path)
        docs = loader.load_and_split()  # Load and split into pages directly
    else:
        raise ValueError("'file_path'must be provided.")
    return docs


# Split documents
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, chunk_overlap=50
    )
    return text_splitter.split_documents(docs)


# Create vectorstore
def create_vectorstore(splits):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()


# Get response using RAG-Fusion
def get_response(vectorstore, user_query):
    llm = ChatOpenAI(temperature=0)
    return run_rag_fusion(vectorstore, user_query, llm)


# Main function
def main():
    with st.sidebar:
        uploaded_files = st.file_uploader(
            "", type=(["pdf"]), accept_multiple_files=True
        )
        process_button = st.button("Process Document(s)")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?")
        ]
    if process_button and (uploaded_files):
        all_docs = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                temp_dir = tempfile.TemporaryDirectory()
                temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())
                docs = load_document(file_path=temp_file_path)
                all_docs.extend(docs)
            splits = split_documents(all_docs)
            st.session_state.vector_store = create_vectorstore(splits)
            st.success("Document(s) processed! You can now ask questions.")
    if "vector_store" in st.session_state:
        user_query = st.chat_input("Type your message here...")
        if user_query is not None and user_query != "":
            response = get_response(st.session_state.vector_store, user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
            source = st.session_state.vector_store.get_relevant_documents(user_query)
            # st.write(source)
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
                    # st.write(f"Source: {source}")
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)
    else:
        st.warning("Please upload PDFs or enter a URL and click 'Process Document(s)'.")


if __name__ == "__main__":
    main()
