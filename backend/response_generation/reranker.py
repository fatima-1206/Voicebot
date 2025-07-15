from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import json
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("config.json", "r") as f:
    config = json.load(f)

# Load the cross-encoder model
reranker_model_id = None
reranker_tokenizer = None
reranker_model = None
reranker_loaded = False

def load_reranker_model():
    global reranker_model_id, reranker_tokenizer, reranker_model, reranker_loaded
    reranker_model_id = config["reranker_model_id"]
    reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_id)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_id).to(device)
    reranker_loaded = True
    print(f"Reranker model loaded: {reranker_model_id}")
    


def rerank_docs(query, docs, top_k=5):
    

    # BGE-Reranker expects [query] </s> document format
    pairs = [f"{query} </s> {doc.page_content}" for doc in docs]

    inputs = reranker_tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        scores = reranker_model(**inputs).logits.view(-1)  # Regression scores

    doc_scores = list(zip(docs, scores.tolist()))
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    reranked_docs = [doc for doc, score in doc_scores[:top_k]]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return reranked_docs


