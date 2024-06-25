import os
import utils
import streamlit as st
from streaming import StreamHandler
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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

load_dotenv()

st.set_page_config(page_title="ThinkAI", page_icon="📄")
st.subheader(
    "Load your PDF, ask questions, and receive answers directly from the document."
)

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter


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


# Generate sub-questions
def generate_sub_questions(question):
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \nThe goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \nGenerate multiple search queries related to: {question} \nOutput (3 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0)
    generate_queries_decomposition = (
        prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n"))
    )
    return generate_queries_decomposition.invoke({"question": question})


# Format question and answer pair
def format_qa_pair(question, answer):
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()


# Answer question using RAG
def answer_question(retriever, question, q_a_pairs=""):
    template = """Here is the question you need to answer:\n --- \n {question} \n --- \nHere is any available background question + answer pairs:\n --- \n {q_a_pairs} \n --- \nHere is additional context relevant to the question: \n --- \n {context} \n --- \nUse the above context and any background question + answer pairs to answer the question: \n {question} \n if the answer to the question is not in context say the answer is not in context"""
    decomposition_prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    rag_chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "q_a_pairs": itemgetter("q_a_pairs"),
        }
        | decomposition_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke({"question": question, "q_a_pairs": q_a_pairs})


def get_response(retriever, question, q_a_pairs=""):
    questions = generate_sub_questions(question)
    answers = []
    for q in questions:
        answer = answer_question(retriever, q, q_a_pairs)
        q_a_pair = format_qa_pair(q, answer)
        q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair
        answers.append(answer)

    final_answer = f"**To answer your question: {question}**\n\n"
    for i, answer in enumerate(answers):
        final_answer += f"**Sub-question {i+1}:** {questions[i]}\n{answer}\n\n"

    return final_answer


# Main function
def main():
    source = None
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
                    if source != None:
                        st.write(source)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)
    else:
        st.warning("Please upload PDFs or enter a URL and click 'Process Document(s)'.")


if __name__ == "__main__":
    main()
