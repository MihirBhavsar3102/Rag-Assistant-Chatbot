from groq import Groq
from backend.core.config import settings
from backend.rag.retriever import get_vectorstore

class ParkerChatbot:
    def __init__(self):
        self.groq = Groq(api_key=settings.GROQ_API_KEY)
        self.vectorstore = get_vectorstore()

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
Hey, I am **Parker**, your Enterprise Knowledge Assistant 🤖. You're chatting in the **{role} Role**.
Respond in a {tone} tone. Use the context below to provide accurate, friendly responses. If the answer is not in the context, do your best or state you don't know based on internal docs.

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
        context = ""
        if self.vectorstore:
            docs = self.vectorstore.similarity_search(query, k=5)
            context = "\n\n".join(doc.page_content for doc in docs)
            
        full_prompt = self._construct_prompt(history, context, query, role)

        response = self.groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": f"You are Parker, an Enterprise Knowledge Assistant. Role: {role}."},
                {"role": "user", "content": full_prompt}
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
        
        if not self.vectorstore:
            return example_questions[:3]
            
        similar = self.vectorstore.similarity_search(user_query, k=1)
        context = similar[0].page_content if similar else ""

        return [
            q for q in example_questions
            if any(word in context.lower() for word in q.lower().split()[:2])
        ][:3] or example_questions[:3]

chatbot_instance = ParkerChatbot()
