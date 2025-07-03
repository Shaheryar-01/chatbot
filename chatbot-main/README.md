# Avanza HR Assistant

Avanza HR Assistant is a simple yet powerful AI chatbot built with **Streamlit** and **LangChain** that helps employees easily find answers to their HR-related questions by interacting with the company employee handbook.

## Features

* 📄 Loads HR policy documents (PDFs) and builds a searchable knowledge base
* 🤖 Chat interface powered by **OpenAI GPT** models
* 🔍 Retrieves relevant document sections using **FAISS** vector search
* 🔎 Multilingual support: Responds in the language of the user's question
* 📱 Easy to deploy with Streamlit for web-based access

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/avanza-hr-assistant.git
cd avanza-hr-assistant
```

### 2. Set Up a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Create a `.env` file in the project root and add your OpenAI API key:

```ini
OPENAI_API_KEY=your_openai_api_key_here
```

---

## How It Works

1. **PDF Download:** The app automatically downloads the employee handbook from a Google Drive link if not present locally.

2. **Document Loading & Vectorstore:**

   * Loads PDFs from the `documents/` folder
   * Splits them into manageable chunks
   * Generates embeddings using **HuggingFace**
   * Builds or loads a **FAISS** index for fast retrieval

3. **Chat with AI:**

   * The user asks a question in the chat.
   * The app retrieves relevant document chunks.
   * GPT model answers based on those chunks only (RAG approach).

4. **Persistent Chat:**

   * Chat history is stored in Streamlit's session state during the session.

---

## Project Structure

```
├── app.py
├── documents/
│   └── 22_Employee Handbook.pdf
├── faiss_index/
├── .env
├── requirements.txt
└── README.md
```

---

---

---

## Tech Stack

* Python 3
* Streamlit
* LangChain
* FAISS
* Hugging Face Transformers
* OpenAI GPT

---

## License

This project is for internal use at **Avanza Solutions**. For external use, please request permission.

---

## Contact

For questions or contributions:

* **Developer:** Shaheryar Baloch
* **Email:** [shaheryar.rahmat@avanzasolutions.com](mailto:shaheryar.rahmat@avanzasolutions.com)
* **Company:** Avanza Solutions
