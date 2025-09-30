import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

text_splitter= RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Use the following website content to answer the question.

    Context:
    {context}

    Question:
    {input}

    Answer:
    """
)

st.set_page_config(page_title="Chat with Website", page_icon="🌐", layout="wide")


st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
        }
        .stChatMessage {
            padding: 10px 15px;
            border-radius: 15px;
            margin-bottom: 10px;
            max-width: 80%;
            word-wrap: break-word;
            color: white;  /* White text */
        }
        .stChatMessage.user {
            background-color: #1f2937;  /* Dark gray */
            margin-left: auto;
            text-align: right;
        }
        .stChatMessage.assistant {
            background-color: #111827;  /* Almost black */
            border: 1px solid #374151;
            text-align: left;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(135deg, #3b82f6, #06b6d4);
            color: white;
        }
        .stButton>button {
            background: linear-gradient(90deg, #06b6d4, #3b82f6);
            color: white;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #2563eb, #0891b2);
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h1 style='text-align:center; color:#2563eb;'>🌐 Chat with Any Website</h1>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

st.sidebar.header("🔗 Website Setup")
url = st.sidebar.text_input("Enter Website URL", placeholder="https://example.com")

if st.sidebar.button("Load Website"):
    if url:
        with st.spinner("🔄 Loading website and creating knowledge base..."):
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                chunks = text_splitter.split_documents(docs)
                vectorstore = Chroma.from_documents(chunks, embeddings)
                st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                st.sidebar.success("✅ Website loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    else:
        st.sidebar.warning("Please enter a website URL")

# Chat Section
st.subheader("💬 Start Chatting")

# Display Chat History
for role, message in st.session_state.chat_history:
    css_class = "user" if role == "user" else "assistant"
    st.markdown(f"<div class='stChatMessage {css_class}'>{message}</div>", unsafe_allow_html=True)

# Input
if user_question := st.chat_input("Ask something about the website..."):
    st.markdown(f"<div class='stChatMessage user'>{user_question}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append(("user", user_question))

    if st.session_state.retriever is None:
        answer = "⚠️ Please load a website first from the sidebar."
    else:
        # Build RAG chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(st.session_state.retriever, document_chain)

        with st.spinner("🤖 Thinking..."):
            response = rag_chain.invoke({"input": user_question})
            answer = response["answer"]

    # Show assistant answer
    st.markdown(f"<div class='stChatMessage assistant'>{answer}</div>", unsafe_allow_html=True)

    st.session_state.chat_history.append(("assistant", answer))
