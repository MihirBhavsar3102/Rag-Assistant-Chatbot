import os
import fitz
import docx2txt
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.rag.retriever import SentenceTransformerEmbeddings
from langchain_core.documents import Document

def extract_text(file_path):
    ext = file_path.split(".")[-1]
    if ext == "pdf":
        doc = fitz.open(file_path)
        return "\n".join(page.get_text() for page in doc)
    elif ext == "docx":
        return docx2txt.process(file_path)
    return ""

def ingest_data(folder_path="data", output_path="faiss_index"):
    if not os.path.exists(folder_path):
        print(f"Data folder {folder_path} not found.")
        return

    docs = []
    for fname in os.listdir(folder_path):
        if fname.endswith((".pdf", ".docx")):
            fpath = os.path.join(folder_path, fname)
            print(f"Processing {fpath}...")
            text = extract_text(fpath)
            docs.append(Document(page_content=text, metadata={"source": fname}))

    if not docs:
        print("No documents found to ingest.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    embeddings = SentenceTransformerEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(output_path)
    print(f"FAISS index saved to '{output_path}/'")

if __name__ == "__main__":
    # Assumes run from project root
    ingest_data("data", "faiss_index")
