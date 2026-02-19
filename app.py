from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# -------- SESSION STORE --------
user_sessions = {}

# -------- LOAD KNOWLEDGE FILE --------
with open("knowledge.txt", "r", encoding="utf-8") as f:
    knowledge_text = f.read()

# Split into chunks
def split_text(text, chunk_size=400):
    words = text.split()
    chunks, current = [], []
    for word in words:
        current.append(word)
        if len(current) >= chunk_size:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

knowledge_chunks = split_text(knowledge_text)

# -------- EMBEDDING MODEL --------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
knowledge_embeddings = embed_model.encode(knowledge_chunks)

# -------- RETRIEVE RELEVANT CONTEXT --------
def retrieve_context(query, top_k=2):
    query_embedding = embed_model.encode([query])[0]
    similarities = np.dot(knowledge_embeddings, query_embedding)
    best_indices = np.argsort(similarities)[-top_k:]
    return "\n".join([knowledge_chunks[i] for i in best_indices])

# -------- ROUTES --------
@app.route("/")
def home():
    return "AI Chatbot with RAG running successfully!"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        message = data.get("message", "")
        session_id = request.remote_addr

        if not message:
            return jsonify({"reply": "No message received."})

        # Track conversation count
        user_sessions[session_id] = user_sessions.get(session_id, 0) + 1
        msg_count = user_sessions[session_id]

        # -------- RAG CONTEXT --------
        context = retrieve_context(message)

        # -------- SYSTEM PROMPT --------
        system_prompt = f"""
You are the official AI Assistant of Algebraa Business Solutions Pvt Ltd.

Always behave like a professional company consultant.

If first interaction, start with:
"Welcome to Algebraa Business Solutions! How can I assist you today?"

Use this company knowledge when relevant:
{context}

Rules:
- Answer ONLY business-related queries
- Speak professionally like a consultant
- Do NOT mention AI models or training data
- Keep responses clear & concise
- Encourage user to explore services

If conversation becomes long, guide them to contact team.

Provide clickable contact info when needed:

Email: <a href="mailto:algebraindia03@gmail.com">algebraindia03@gmail.com</a>
Phone: <a href="tel:+919442228766">+91-9442228766</a>
"""

        # After 6 messages → add support suggestion
        if msg_count >= 6:
            message += "\nPlease also guide me to contact your team."

        # -------- OPENROUTER CALL --------
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ]
            }
        )

        result = response.json()
        reply = result["choices"][0]["message"]["content"]

        return jsonify({"reply": reply})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"reply": "Server error. Please try again."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
