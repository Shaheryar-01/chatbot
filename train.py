# Updated train_and_pickle.py using pdfplumber
import pickle
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Extract text using pdfplumber
def load_text_with_pdfplumber(pdf_path):
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                documents.append(Document(page_content=text, metadata={"page": i + 1}))
    return documents

# Load and split document
pdf_path = "22_Employee Handbook.pdf"
docs = load_text_with_pdfplumber(pdf_path)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(docs)


# Create embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store
vectorstore = FAISS.from_documents(chunks, embedding=embedding)

# Save retriever to disk using pickle
with open("file.pkl", "wb") as f:
    pickle.dump(vectorstore, f)

print("✅ Document trained with pdfplumber and saved to file.pkl")