import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
from collections import deque

# Load vector database
def load_vector_db():
    with open("vector_db.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    index = faiss.read_index("faiss_index")
    embeddings = np.load("embeddings.npy")
    return data["chunks"], index, embeddings

# Retrieve relevant chunks
def retrieve_chunks(query: str, chunks: list, index, embeddings, model, top_k: int = 3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# Generate answer using ollama locally
def generate_answer_with_llm(query: str, retrieved_chunks: list, chat_history: deque) -> str:
    # Short-term memory: Include last 3 queries and responses
    history_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in list(chat_history)[-3:]])
    context = history_context + "\n" + "\n".join(f"[{chunk['type']}][Page {chunk['page']}]: {chunk['text']}" for chunk in retrieved_chunks)
    prompt = f"Based on the following context, provide a precise answer in the same language as the query (Bengali or English). Extract the answer directly from the context, prioritizing exact matches for facts or questions. If no clear answer is found, say 'দুঃখিত, উত্তর পাওয়া যায়নি।' or 'Sorry, no answer found.' Query: {query}\nContext:\n{context}"
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "tinyllama",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "max_tokens": 950}
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60, stream=False)
        if response.status_code == 200:
            response_text = response.text.strip()
            if '"response":"' in response_text:
                # Extract response from JSON
                return response_text.split('"response":"')[1].split('"')[0].strip()
            elif "response" in response_text:
                # Handle potential NDJSON or alternative format
                import json
                data = json.loads(response_text) if response_text.startswith('{') else {"response": response_text}
                return data.get("response", "দুঃখিত, উত্তর পাওয়া যায়নি।").strip()
            return "দুঃখিত, উত্তর পাওয়া যায়নি।"
        else:
            print(f"Ollama error: Status code {response.status_code}")
            return f"দুঃখিত, সার্ভার ত্রুটি (কোড: {response.status_code})।"
    except requests.exceptions.RequestException as e:
        print(f"Ollama connection failed: {str(e)}")
        return f"দুঃখিত, সার্ভারের সাথে সংযোগ বিফল ({str(e)})。"

if __name__ == "__main__":
    model = SentenceTransformer("xlm-roberta-base")
    chunks, index, embeddings = load_vector_db()
    chat_history = deque(maxlen=3)  # Short-term memory for last 3 interactions
    
    print("চ্যাটবট চালু হয়েছে! প্রশ্ন জিজ্ঞাসা করুন (প্রশ্ন শেষ করতে 'quit' লিখুন):")
    while True:
        query = input("প্রশ্ন: ").strip()
        if query.lower() in ["quit", "exit", "বন্ধ"]:
            print("চ্যাটবট বন্ধ করা হলো। ধন্যবাদ!")
            break
        if not query:
            print("কোনো প্রশ্ন দেয়নি! একটি প্রশ্ন দিন।")
            continue
        retrieved = retrieve_chunks(query, chunks, index, embeddings, model)
        answer = generate_answer_with_llm(query, retrieved, chat_history)
        chat_history.append((query, answer))  # Update short-term memory
        print(f"প্রশ্ন: {query}")
        print(f"উত্তর: {answer}")
        print(f"সম্পর্কিত টুকরো: {[c['text'][:50] + '...' for c in retrieved]}")
        print("-" * 50)