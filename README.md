# Enterprise Knowledge Assistant

A secure, scalable RAG (Retrieval-Augmented Generation) application built for internal company use. It answers HR policy questions, IT SOP queries, and other internal document questions based on your data.

## 🧱 Architecture

- **Frontend:** Streamlit UI
- **Backend:** FastAPI
- **LLM:** Groq (can be configured for OpenAI/Azure)
- **Vector DB:** FAISS
- **Embeddings:** Sentence-Transformers

## 📁 Project Structure

```text
├── backend/          # FastAPI server and RAG logic (retriever, chatbot engine)
├── frontend/         # Streamlit User Interface
├── data/             # Drop your source documents (.pdf, .docx) here
├── faiss_index/      # Generated vector database (created automatically)
├── .env              # Environment variables (API keys)
└── ...
```

## 🚀 Getting Started

### 1. Setup Virtual Environment
First, create and activate a Python virtual environment:
```powershell
# Create environment
python -m venv venv

# Activate environment (Windows)
.\venv\Scripts\activate
```

### 2. Install Dependencies
Install all required packages from the backend directory:
```powershell
pip install -r backend/requirements_free.txt
```

### 3. Environment Variables
Create a `.env` file in the root of the project and add your API keys:
```env
GROQ_API_KEY=your_groq_api_key_here
# API_URL=http://localhost:8000  (Optional, defaults to this if not set)
```

### 4. Ingest Documents
1. Place any `.pdf` or `.docx` files you want the assistant to learn from into the `data/` folder.
2. The documents will be processed and indexed automatically when you trigger the ingestion process.

## 🏃‍♂️ Running the Application

To run the application, you need to start **both** the backend API and the frontend UI in separate terminal windows. Make sure your virtual environment is activated in both terminals!

### Start the FastAPI Backend (Terminal 1)
```powershell
uvicorn backend.main:app --reload
```
*The API will be available at `http://127.0.0.1:8000` (Visit `/docs` for the interactive Swagger UI).*

### Start the Streamlit Frontend (Terminal 2)
```powershell
streamlit run frontend/app.py
```
*This will open the web interface in your default browser.*
