import re
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict

# Load and clean text
def load_and_clean_text(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Remove page headers and extra dashes
    text = re.sub(r'page -\d+\n|---------\n', '', text)
    lines = text.split('\n')
    chunks = []

    i = 1
    current_paragraph = []
    for line in lines:
        line = line.strip()
        if not line:
            if current_paragraph:
                chunks.append({"id": f"para_{i}", "text": " ".join(current_paragraph), "type": "paragraph", "page": i})
                i += 1
                current_paragraph = []
            continue

        # Detect QA pairs (e.g., "৮০। ... (খ) ...")
        qa_match = re.match(r'(\d+।\s+.*?\?)[\s(]([ক-ঘ])\)\s+.*', line)
        if qa_match:
            question = qa_match.group(1).strip()
            answer = re.search(r'\((ক|খ|গ|ঘ)\)\s+(.+)', line)
            if answer:
                chunks.append({"id": f"qa_{i}", "text": f"{question} Answer: {answer.group(2)}", "type": "qa", "page": i, "option": answer.group(1)})
                i += 1
            continue

        # Detect word-meaning pairs (e.g., "মঞ্জরী - কিশলয়যুক্ত ...")
        word_meaning_match = re.match(r'(.+?)\s*-\s*(.+)', line)
        if word_meaning_match:
            word, meaning = word_meaning_match.groups()
            chunks.append({"id": f"word_{i}", "text": f"{word}: {meaning}", "type": "word_meaning", "page": i})
            i += 1
            continue

        # Accumulate lines into paragraphs
        current_paragraph.append(line)
    
    if current_paragraph:
        chunks.append({"id": f"para_{i}", "text": " ".join(current_paragraph), "type": "paragraph", "page": i})

    return chunks

# Create vector embeddings
def create_vector_db(chunks: List[Dict], model_name: str = "xlm-roberta-base"):
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    # Initialize FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    # Save metadata and index
    data = {"chunks": chunks, "index": index.ntotal}
    with open("vector_db.json", "w", encoding="utf-8") as f:
        json.dump({"chunks": [{"id": c["id"], "text": c["text"], "type": c["type"], "page": c["page"], "option": c.get("option")} for c in chunks]}, f, ensure_ascii=False, indent=2)
    faiss.write_index(index, "faiss_index")
    np.save("embeddings.npy", embeddings)
    return embeddings, index

if __name__ == "__main__":
    chunks = load_and_clean_text("Data/cleaned_bangla_qa.txt")
    embeddings, index = create_vector_db(chunks)
    print(f"Created {len(chunks)} chunks and vector database with {index.ntotal} vectors.")
