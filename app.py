from pathlib import Path
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import pickle
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")


def resolve_local_embedding_model_path():
    candidates = [
        BASE_DIR / "embedding_model_local",
        BASE_DIR / "embedding_model_local" / "model",
    ]

    for candidate in candidates:
        if candidate.exists() and (candidate / "config.json").exists():
            return candidate
    return None


def local_sentiment_fallback(text: str):
    normalized = text.lower()
    negative_terms = [
        "angry", "frustrated", "bad", "terrible", "hate", "ridiculous",
        "cancel", "problem", "issue", "wrong", "broken", "unhappy",
        "delay", "late", "refund", "angry", "upset", "annoyed",
    ]
    positive_terms = [
        "thanks", "thank you", "happy", "great", "good", "love",
        "excellent", "nice", "satisfied", "helpful",
    ]

    negative_hits = sum(1 for term in negative_terms if term in normalized)
    positive_hits = sum(1 for term in positive_terms if term in normalized)

    if negative_hits > positive_hits:
        score = 0.85 if negative_hits >= 2 else 0.65
        return {"sentiment": "negative", "confidence": round(score, 4)}
    if positive_hits > negative_hits:
        score = 0.82 if positive_hits >= 2 else 0.62
        return {"sentiment": "positive", "confidence": round(score, 4)}

    return {"sentiment": "neutral", "confidence": 0.5}


intent_model_path = BASE_DIR / "model_final"
intent_tokenizer = AutoTokenizer.from_pretrained(str(intent_model_path))
intent_model = AutoModelForSequenceClassification.from_pretrained(str(intent_model_path))
intent_model.to(device)
intent_model.eval()

with open(intent_model_path / "label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

sentiment_analyzer = None

embedding_model_path = resolve_local_embedding_model_path()
if embedding_model_path is None:
    raise RuntimeError("Missing local embedding model folder: embedding_model_local")

embedding_model = SentenceTransformer(str(embedding_model_path))
train_df = pd.read_csv(BASE_DIR / "data" / "train_data.csv")
instruction_embeddings = np.load(BASE_DIR / "data" / "instruction_embeddings.npy")

print("All models loaded successfully")


def predict_intent(text: str):
    inputs = intent_tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = intent_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)

    predicted_id = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][predicted_id].item()
    predicted_intent = label_encoder.inverse_transform([predicted_id])[0]

    return {"intent": predicted_intent, "confidence": round(confidence, 4)}


def get_sentiment(text: str):
    if sentiment_analyzer is None:
        return local_sentiment_fallback(text)

    result = sentiment_analyzer(text)[0]
    return {
        "sentiment": result["label"].lower(),
        "confidence": round(result["score"], 4),
    }


def find_similar_response(new_message: str, predicted_intent: str, top_k: int = 1):
    new_embedding = embedding_model.encode([new_message])
    similarities = cosine_similarity(new_embedding, instruction_embeddings)[0]
    sorted_indices = similarities.argsort()[::-1]

    matched_results = []
    for idx in sorted_indices:
        row = train_df.iloc[idx]
        if row["intent"] == predicted_intent:
            matched_results.append({
                "matched_instruction": row["instruction"],
                "suggested_response": row["response"],
                "similarity_score": round(float(similarities[idx]), 4),
            })
        if len(matched_results) >= top_k:
            break

    if not matched_results:
        for idx in sorted_indices[:top_k]:
            row = train_df.iloc[idx]
            matched_results.append({
                "matched_instruction": row["instruction"],
                "suggested_response": row["response"],
                "similarity_score": round(float(similarities[idx]), 4),
            })

    return matched_results


def process_customer_message(message: str, intent_confidence_threshold: float = 0.5):
    intent_result = predict_intent(message)
    sentiment_result = get_sentiment(message)

    low_confidence_intent = intent_result["confidence"] < intent_confidence_threshold

    if low_confidence_intent:
        return {
            "original_message": message,
            "predicted_intent": "unclear / general_comment",
            "intent_confidence": intent_result["confidence"],
            "sentiment": sentiment_result["sentiment"],
            "sentiment_confidence": sentiment_result["confidence"],
            "priority": "low",
            "note": "Message does not clearly match a known support intent (e.g. a thank-you or general comment). No automatic response suggested - route to a general acknowledgment or human review.",
            "suggested_response": None,
            "similarity_score": None,
        }

    similar_results = find_similar_response(
        message, predicted_intent=intent_result["intent"], top_k=1,
    )
    best_match = similar_results[0]

    if sentiment_result["sentiment"] == "negative" and sentiment_result["confidence"] > 0.8:
        priority = "high"
        note = "Customer appears clearly upset - recommend reviewing the response before sending and prioritizing a quick reply"
    elif sentiment_result["sentiment"] == "negative":
        priority = "medium"
        note = "Customer seems dissatisfied - please review the suggested response before sending"
    else:
        priority = "normal"
        note = "The suggested response can be sent directly after a quick review"

    return {
        "original_message": message,
        "predicted_intent": intent_result["intent"],
        "intent_confidence": intent_result["confidence"],
        "sentiment": sentiment_result["sentiment"],
        "sentiment_confidence": sentiment_result["confidence"],
        "priority": priority,
        "note": note,
        "suggested_response": best_match["suggested_response"],
        "similarity_score": best_match["similarity_score"],
    }


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Please send a JSON body with a 'message' field"}), 400

    message = data["message"]

    if not isinstance(message, str) or len(message.strip()) == 0:
        return jsonify({"error": "The message must be a non-empty string"}), 400

    try:
        result = process_customer_message(message)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred while processing the message: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "TicketMind API is running"}), 200


if __name__ == "__main__":
    app.run(port=5000, debug=False)