# src/ingest.py

import os
import fitz
import docx2txt
import tempfile
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

def extract_text(file_path):
    ext = file_path.split(".")[-1]
    if ext == "pdf":
        doc = fitz.open(file_path)
        return "\n".join(page.get_text() for page in doc)
    elif ext == "docx":
        return docx2txt.process(file_path)
    return ""

def main():
    folder = "data"
    docs = []
    for fname in os.listdir(folder):
        if fname.endswith((".pdf", ".docx")):
            fpath = os.path.join(folder, fname)
            text = extract_text(fpath)
            docs.append(Document(page_content=text))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    embeddings = SentenceTransformerEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local("faiss_index")
    print("FAISS index saved to 'faiss_index/'")

if __name__ == "__main__":
    main()
