from flask import Flask, render_template, request
from collections import deque
import json
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer, util
from together import Together
from config import API_KEY

from rag import load_vector_db, retrieve_chunks,save_evaluation  # Keep this for clean reuse

app = Flask(__name__)

# Lazy load
model = None
chunks, index, embeddings = None, None, None
chat_history = deque(maxlen=3)

# Mode toggle: use_local=True -> Ollama, False -> Together API
USE_LOCAL = False  # ⬅️ Change this to False to switch to Together API

def initialize_resources():
    global model, chunks, index, embeddings
    if model is None:
        try:
            model = SentenceTransformer("xlm-roberta-base")  # May need lighter model if memory fails
        except Exception as e:
            print(f"Error loading model: {e}. Using lighter model 'paraphrase-multilingual-MiniLM-L12-v2'.")
            model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    if chunks is None:
        chunks, index, embeddings = load_vector_db()

def generate_answer_local(query: str, retrieved_chunks: list, chat_history: deque) -> str:
    history_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history])
    context = history_context + "\n" + "\n".join(f"[{c['type']}][Page {c['page']}]: {c['text']}" for c in retrieved_chunks)
    prompt = f"Based on the following context, answer the query in Bengali or English. If unsure, say 'দুঃখিত, উত্তর পাওয়া যায়নি।'. Query: {query}\nContext:\n{context}"
    url = "http://localhost:11434/api/generate"
    payload = {"model": "llama3", "prompt": prompt, "stream": False}
    try:
        res = requests.post(url, json=payload, timeout=60)
        if res.status_code == 200:
            response = res.json()
            return response.get("response", "দুঃখিত, উত্তর পাওয়া যায়নি।")
        else:
            return f"দুঃখিত, সার্ভার ত্রুটি: {res.status_code}"
    except Exception as e:
        return f"দুঃখিত, ব্যর্থ ({str(e)})"

def generate_answer_api(query: str, retrieved_chunks: list, chat_history: deque) -> str:
    context = "\n".join(f"[{c['type']}][Page {c['page']}]: {c['text']}" for c in retrieved_chunks)
    prompt = f"Given this context, answer in Bengali or English. Query: {query}\nContext:\n{context}"
    client = Together(api_key=API_KEY)
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"দুঃখিত, ব্যর্থ ({str(e)})"

def evaluate_relevance(query_embedding, retrieved_embeddings):
    similarities = util.cos_sim(query_embedding, retrieved_embeddings).squeeze(0)
    return similarities.mean().item()

def check_groundedness(answer, retrieved_chunks):
    return any(
        answer in chunk["text"] or any(word in chunk["text"] for word in answer.split())
        for chunk in retrieved_chunks
    )

@app.route("/", methods=["GET", "POST"])
def index():
    initialize_resources()
    answer = ""
    query = ""
    relevance = None
    groundedness = None
    retrieved_preview = []
    mode = "Local Ollama" if USE_LOCAL else "Together API"  # default mode

    if request.method == "POST":
        query = request.form["query"].strip()
        # Get mode from form, fallback to default
        mode_choice = request.form.get("mode", "local")  # "local" or "api"
        use_local = (mode_choice == "local")

        if not query:
            answer = "Please enter a question."
        else:
            retrieved = retrieve_chunks(query, chunks, index, embeddings, model)
            if use_local:
                answer = generate_answer_local(query, retrieved, chat_history)
                mode = "Local Ollama"
            else:
                answer = generate_answer_api(query, retrieved, chat_history)
                mode = "Together API"
            chat_history.append((query, answer))

            query_emb = model.encode([query], convert_to_tensor=True)
            retrieved_emb = model.encode([c["text"] for c in retrieved], convert_to_tensor=True)

            relevance = evaluate_relevance(query_emb, retrieved_emb)
            groundedness = check_groundedness(answer, retrieved)

            retrieved_preview = [c["text"][:80] + "..." for c in retrieved]
            save_evaluation(
                query=query,
                answer=answer,
                relevance=f"{relevance:.3f}",
                groundedness="Yes" if groundedness else "No",
                retrieved_chunks=retrieved_preview
            )

            return render_template(
                "index.html",
                query=query,
                answer=answer,
                retrieved_chunks=retrieved_preview,
                relevance=f"{relevance:.3f}",
                groundedness="Yes" if groundedness else "No",
                mode=mode,
                selected_mode=mode_choice
            )

    return render_template(
        "index.html",
        query=query,
        answer=answer,
        retrieved_chunks=retrieved_preview,
        relevance=relevance,
        groundedness=groundedness,
        mode=mode,
        selected_mode="local" if USE_LOCAL else "api"
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
