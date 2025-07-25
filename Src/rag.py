import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from together import Together
from config import API_KEY
import csv
import os
from sentence_transformers import util
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

# Generate answer using Together API
def generate_answer_with_llm(query: str, retrieved_chunks: list) -> str:
    context = "\n".join(f"[{chunk['type']}][Page {chunk['page']}]: {chunk['text']}" + (f" [Option: {chunk.get('option', '')}]" if chunk.get("option") else "") for chunk in retrieved_chunks)
    prompt = f"Based on the following context, provide a precise answer in Bengali. The context includes paragraphs, QA pairs with options, and word-meaning pairs. Prioritize exact matches for professions (e.g., 'কী করতেন' should return 'ওকালতি' if present). For QA pairs, select the correct option. If no clear answer, say 'দুঃখিত, উত্তর পাওয়া যায়নি।'. Query: {query}\nContext:\n{context}"
    
    client = Together(api_key=API_KEY)# Replace with your key
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        result = response.choices[0].message.content.strip()
        return result if result else "দুঃখিত, উত্তর পাওয়া যায়নি।"
    except Exception as e:
        return f"দুঃখিত, সার্ভার ত্রুটি ({str(e)})。"

if __name__ == "__main__":
    model = SentenceTransformer("xlm-roberta-base")
    chunks, index, embeddings = load_vector_db()
    
    print("চ্যাটবট চালু হয়েছে! প্রশ্ন জিজ্ঞাসা করুন (প্রশ্ন শেষ করতে 'quit' লিখুন):")
    while True:
        query = input("প্রশ্ন: ").strip()
        if query.lower() in ["quit", "exit"]:
            print("চ্যাটবট বন্ধ করা হলো। ধন্যবাদ!")
            break
        if not query:
            print("কোনো প্রশ্ন দেয়নি! একটি প্রশ্ন দিন।")
            continue
        retrieved = retrieve_chunks(query, chunks, index, embeddings, model)
        answer = generate_answer_with_llm(query, retrieved)
        print(f"উত্তর: {answer}")
        print(f"সম্পর্কিত টুকরো: {[c['text'][:50] + '...' for c in retrieved]}")
        print("-" * 50)
        
        


def evaluate_relevance(query_embedding, retrieved_embeddings):
    """
    Compute average cosine similarity between query and retrieved chunks.
    """
    similarities = util.cos_sim(query_embedding, retrieved_embeddings).squeeze(0)
    avg_similarity = similarities.mean().item()
    return avg_similarity

def check_groundedness(answer, retrieved_chunks):
    """
    Check if answer text overlaps significantly with any retrieved chunk.
    Simple heuristic: word overlap or substring check.
    """
    for chunk in retrieved_chunks:
        if chunk['text'] in answer or any(word in chunk['text'] for word in answer.split()):
            return True
    return False



def save_evaluation(query, answer, relevance, groundedness, retrieved_chunks):
    file_exists = os.path.isfile("rag_eval_log.csv")
    with open("evaluation/evaluation_log.csv", mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Query", "Answer", "Relevance", "Groundedness", "RetrievedChunks"])
        writer.writerow([
            query,
            answer,
            relevance,
            groundedness,
            " | ".join(retrieved_chunks)
        ])
