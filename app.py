# app.py
import streamlit as st
import pickle
from openai import OpenAI
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Avanza HR Assistant",
    page_icon="avanza.png",  # Replace with your desired favicon/logo
    layout="centered"
)

# --- Display logo and title ---
st.image("avanza_solutions.png", width=200)
st.title("Avanza HR Assistant")
st.caption("🤖 HR made simple. Ask me anything from the Employee Handbook!")

# --- Load Vectorstore ---
@st.cache_resource
def load_vectorstore():
    url = "https://drive.google.com/uc?export=download&id=1LRcof-2qDV0V5FeRPPJx6Zdruw23BOOq"
    local_path = "file.pkl"

    if not os.path.exists(local_path):
        with st.spinner("📦 Downloading vectorstore..."):
            response = requests.get(url)
            with open(local_path, "wb") as f:
                f.write(response.content)


    with open("file.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    return vectorstore.as_retriever(search_kwargs={"k": 20})  # Reduced k for relevance

retriever = load_vectorstore()

# --- OpenAI Client Init ---
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=api_key,
)

# --- Question Answering Logic ---
def ask_question(query):
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Construct prompt with memory and multilingual instruction
    history = [
        {"role": "system", "content": "You are a helpful HR assistant. Only use the context from the company employee handbook to answer. Do not guess or add extra info."},
        {"role": "system", "content": f"Document Context:\n{context}"},
        {"role": "system", "content": "Always respond in the same language as the user's question."}
    ]

    # Add chat history (limit to last 3 pairs to stay within token limit)
    for msg in st.session_state.messages[-6:]:
        history.append({"role": msg["role"], "content": msg["content"]})

    # Add the current question
    history.append({"role": "user", "content": query})

    # Call OpenAI model
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=history,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# --- Session State for Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display past messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User Input ---
user_query = st.chat_input("Ask a question about HR policies...")

if user_query:
    # Show user message
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.spinner("Thinking..."):
        bot_reply = ask_question(user_query)

    # Show assistant message
    st.chat_message("assistant").markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
