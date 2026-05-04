import streamlit as st
import requests
import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# === Streamlit UI ===
st.set_page_config(page_title="Enterprise Knowledge Assistant", layout="wide")
st.title("Enterprise Knowledge Assistant")

# Track role selection
if "selected_role" not in st.session_state:
    st.session_state.selected_role = "General"

new_role = st.sidebar.selectbox("🧑 Select your role", ["General", "HR", "Tech", "Sales", "Intern"])

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
        "General": f"{role_emoji} You're now chatting in **General mode**.",
        "HR": f"{role_emoji} Assistant has switched to the **HR Role**.",
        "Tech": f"{role_emoji} Assistant has switched to the **Technical Role**.",
        "Sales": f"{role_emoji} Assistant has switched to the **Sales Role**.",
        "Intern": f"{role_emoji} You’re now chatting as an **Intern**."
    }

    announcement = role_message_map.get(new_role)
    if "messages" in st.session_state:
        st.session_state.messages.append({"role": "assistant", "content": announcement})
    st.session_state.selected_role = new_role

role = st.session_state.selected_role

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    intro = f"Hey 👋, I’m your **Enterprise Knowledge Assistant**.\nYou're currently chatting in the **{role} Role**.\nAsk me anything based on our internal docs!"
    st.session_state.messages.append({"role": "assistant", "content": intro})

if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input query
query = st.chat_input("Ask your question here...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Send request to FastAPI backend
                payload = {
                    "query": query,
                    "role": role,
                    "history": [msg for msg in st.session_state.messages[:-1]] # exclude current query
                }
                res = requests.post(f"{API_URL}/chat", json=payload)
                res.raise_for_status()
                data = res.json()
                response_text = data.get("response", "No response received.")
                st.session_state.suggestions = data.get("suggestions", [])
                
                st.markdown(response_text)
            except Exception as e:
                response_text = f"❌ Error contacting backend: {e}"
                st.error(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.rerun()

# Suggested Questions from FastAPI
with st.sidebar:
    st.markdown("### 💡 Suggested Questions")
    if st.session_state.suggestions:
        for q in st.session_state.suggestions:
            st.markdown(f"- {q}")
    else:
        st.markdown("_Ask something to see relevant suggestions._")
