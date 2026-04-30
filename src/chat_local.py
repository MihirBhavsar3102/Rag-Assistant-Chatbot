import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

# === Load GROQ API Key ===
load_dotenv()
# groq_api_key = ""
# if not groq_api_key:
#     raise ValueError("GROQ_API_KEY not found in .env")

# === Embedding Wrapper ===
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

# === Parker Chatbot ===
class ParkerChatbot:
    def __init__(self, groq_key):
        self.groq = Groq(api_key=groq_key)
        self.embeddings = SentenceTransformerEmbeddings()
        self.vectorstore = FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)

    def _construct_prompt(self, history, context, query, role):
        tone_map = {
            "General": "neutral and helpful",
            "HR": "formal and policy-aware",
            "Tech": "detailed and technical",
            "Sales": "persuasive and concise",
            "Intern": "simple and beginner-friendly"
        }
        tone = tone_map.get(role, "helpful")

        intro = f"""
Hey, I am **Parker**, your RAG-based assistant 🤖. You're chatting in the **{role} Role**.
Respond in a {tone} tone. Use the context below to provide accurate, friendly responses.

📄 Context:
{context}

💬 Chat history:
"""
        chat_history = ""
        for msg in history:
            role_prefix = "User" if msg["role"] == "user" else "Parker"
            chat_history += f"{role_prefix}: {msg['content']}\n"

        prompt = f"{intro}\n{chat_history}\nUser: {query}\nParker:"
        return prompt

    def query(self, query, role, history):
        docs = self.vectorstore.similarity_search(query, k=5)
        context = "\n\n".join(doc.page_content for doc in docs)
        full_prompt = self._construct_prompt(history, context, query, role)

        response = self.groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": f"You are Parker, a smart assistant who adapts tone and language based on the user’s selected role: {role}."},
                {"role": "user", "content": full_prompt}
            ]
        )

        return response.choices[0].message.content

    def suggest_questions(self, user_query):
        """Generate relevant suggested questions from the document using FAISS"""
        example_questions = [
            "What are the key policies?",
            "Summarize the leave section.",
            "Where is the contact info?",
            "What does the document say about remote work?",
            "What is the latest version or update?",
            "What’s the procedure for onboarding?"
        ]
        similar = self.vectorstore.similarity_search(user_query, k=1)
        context = similar[0].page_content if similar else ""

        return [
            q for q in example_questions
            if any(word in context.lower() for word in q.lower().split()[:2])
        ][:3] or example_questions[:3]

# === Streamlit UI ===
st.set_page_config(page_title="Parker - Contextual Assistant", layout="wide")
st.title("💬 Parker - RAG Assistant Chatbot")

# Track role selection
if "selected_role" not in st.session_state:
    st.session_state.selected_role = "General"

new_role = st.sidebar.selectbox("🧑 Select your role", ["General", "HR", "Tech", "Sales", "Intern"])
chatbot = ParkerChatbot(groq_api_key)

# Detect role change and announce it
if new_role != st.session_state.selected_role:
    role_emoji = {
        "General": "👤",
        "HR": "🧑‍💼",
        "Tech": "🧑‍💻",
        "Sales": "💼",
        "Intern": "🎓"
    }.get(new_role, "👤")

    role_message_map = {
        "General": f"{role_emoji} You're now chatting in **General mode**. Parker will maintain a neutral and helpful tone.",
        "HR": f"{role_emoji} Parker has switched to the **HR Role**. Expect formal responses focused on policies and procedures.",
        "Tech": f"{role_emoji} Parker has switched to the **Technical Role**. Answers will now include detailed, technical explanations.",
        "Sales": f"{role_emoji} Parker has switched to the **Sales Role**. Responses will be concise and focused on customer-oriented insights.",
        "Intern": f"{role_emoji} You’re now chatting as an **Intern**. Parker will simplify content and provide beginner-friendly support."
    }

    announcement = role_message_map.get(new_role)
    st.session_state.messages.append({"role": "assistant", "content": announcement})
    st.session_state.selected_role = new_role

role = st.session_state.selected_role

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    intro = f"Hey 👋, I’m **Parker**, your RAG-powered assistant.\nYou're currently chatting in the **{role} Role**.\nAsk me anything from your uploaded documents!"
    st.session_state.messages.append({"role": "assistant", "content": intro})

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input query
query = st.chat_input("Ask Parker your question here...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Parker is thinking..."):
            response = chatbot.query(query, role, st.session_state.messages)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# Suggested Questions from FAISS
with st.sidebar:
    st.markdown("### 💡 Suggested Questions")
    if st.session_state.messages:
        last_user_input = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), "")
        suggestions = chatbot.suggest_questions(last_user_input)
        for q in suggestions:
            st.markdown(f"- {q}")
    else:
        st.markdown("_Ask something to see relevant suggestions._")
