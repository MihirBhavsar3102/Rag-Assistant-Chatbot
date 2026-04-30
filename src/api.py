import os
from flask import Flask, request, jsonify, render_template_string, session
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

# === Load Environment ===
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

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
        prompt = self._construct_prompt(history, context, query, role)

        response = self.groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": f"You are Parker, a smart assistant who adapts tone and language based on the user’s selected role: {role}."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    def suggest_questions(self, user_query):
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


# === Flask App ===
app = Flask(__name__)
app.secret_key = "supersecretkey"
chatbot = ParkerChatbot(groq_api_key)

# === HTML Template ===
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Parker - RAG Chatbot</title>
    <style>
        body { font-family: Arial; padding: 20px; max-width: 700px; margin: auto; background-color: #f5f5f5; }
        .chat-box { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px #ddd; }
        .user { color: #222; font-weight: bold; }
        .assistant { color: #007BFF; font-weight: bold; }
        .msg { margin-bottom: 12px; }
        select, input[type=text] { padding: 8px; width: 100%; margin: 10px 0; }
        .suggestions { margin-top: 20px; }
    </style>
</head>
<body>
    <div class="chat-box">
        <h2>🤖 Parker - RAG Assistant</h2>
        <form method="post">
            <label for="role">Select your role:</label>
            <select name="role" id="role">
                {% for r in roles %}
                    <option value="{{r}}" {% if r==role %}selected{% endif %}>{{r}}</option>
                {% endfor %}
            </select>

            <label for="query">Ask Parker:</label>
            <input type="text" name="query" id="query" required>
            <button type="submit">Send</button>
        </form>

        <hr>
        {% for msg in history %}
            <div class="msg">
                <span class="{{msg.role}}">{{msg.role.capitalize()}}:</span>
                <span>{{msg.content}}</span>
            </div>
        {% endfor %}

        {% if suggestions %}
            <div class="suggestions">
                <strong>💡 Suggested Questions:</strong>
                <ul>
                    {% for q in suggestions %}
                        <li>{{ q }}</li>
                    {% endfor %}
                </ul>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

# === Routes ===
@app.route("/", methods=["GET", "POST"])
def index():
    if "messages" not in session:
        session["messages"] = [{"role": "assistant", "content": "Hey 👋, I’m Parker, your RAG assistant. Ask me anything based on the document knowledge base!"}]
    if "role" not in session:
        session["role"] = "General"

    if request.method == "POST":
        query = request.form["query"]
        role = request.form["role"]
        session["role"] = role

        session["messages"].append({"role": "user", "content": query})
        response = chatbot.query(query, role, session["messages"])
        session["messages"].append({"role": "assistant", "content": response})

        suggestions = chatbot.suggest_questions(query)
    else:
        suggestions = []

    return render_template_string(
        HTML_PAGE,
        history=session["messages"],
        suggestions=suggestions,
        role=session["role"],
        roles=["General", "HR", "Tech", "Sales", "Intern"]
    )


# === Run ===
if __name__ == "__main__":
    app.run(debug=True)
