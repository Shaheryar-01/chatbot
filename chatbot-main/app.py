# app.py
import os
import streamlit as st
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Avanza HR Assistant",
    page_icon="avanza.png",
    layout="centered"
)

# --- Display logo and title ---
st.image("avanza_solutions.png", width=200)
st.title("Avanza HR Assistant")
st.caption("🤖 HR made simple. Ask me anything from the Employee Handbook!")

# --- Step 1: Download PDF from Google Drive ---
PDF_URL = "https://drive.google.com/uc?export=download&id=1LRcof-2qDV0V5FeRPPJx6Zdruw23BOOq"
PDF_PATH = "22_Employee Handbook.pdf"

if not os.path.exists(PDF_PATH):
    with st.spinner("📥 Downloading employee handbook..."):
        response = requests.get(PDF_URL)
        with open(PDF_PATH, "wb") as f:
            f.write(response.content)

# --- Step 2: Build Vectorstore ---
@st.cache_resource
def load_vectorstore():
    print("📦 Loading vectorstore...")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Check if FAISS index exists
    if os.path.exists("faiss_index/index.faiss") and os.path.exists("faiss_index/index.pkl"):
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        print("✅ Loaded existing FAISS index")
    else:
        print("⚙️ No existing index found, building now...")
        all_docs = []
        folder_path = "documents"
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                full_path = os.path.join(folder_path, filename)
                loader = PyPDFLoader(full_path)
                documents = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.split_documents(documents)
                all_docs.extend(chunks)

        vectorstore = FAISS.from_documents(all_docs, embedding=embeddings)
        vectorstore.save_local("faiss_index")
        print("✅ New FAISS index built and saved")

    return vectorstore.as_retriever(search_kwargs={"k": 20})


retriever = load_vectorstore()

# --- Step 3: Load QA Chain ---
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0.7, openai_api_key=api_key)
qa_chain = load_qa_chain(llm, chain_type="stuff")

# --- Question Answering Logic ---
def ask_question(query):
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    history = [
        {"role": "system", "content": "You are a helpful HR assistant. Only use the context from the company employee handbook to answer. Do not guess or add extra info."},
        {"role": "system", "content": f"Document Context:\n{context}"},
        {"role": "system", "content": "Always respond in the same language as the user's question."}
    ]

    # Add recent chat history (last 3 turns)
    for msg in st.session_state.messages[-6:]:
        history.append({"role": msg["role"], "content": msg["content"]})

    history.append({"role": "user", "content": query})

    try:
        response = llm.invoke(history)
        return response.content.strip()

    except Exception as e:
        return f"❌ Error: {str(e)}"

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question about HR policies...")

if user_query:
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.spinner("Thinking..."):
        bot_reply = ask_question(user_query)

    st.chat_message("assistant").markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
