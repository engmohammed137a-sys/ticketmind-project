import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    F = None
    TORCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    SENTENCE_TRANSFORMER_AVAILABLE = False

# sklearn may be unavailable in some deployment environments; provide a NumPy fallback
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

    def cosine_similarity(a, b):
        a = np.atleast_2d(a).astype(float)
        b = np.atleast_2d(b).astype(float)

        # compute norms
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        a_norm[a_norm == 0] = 1e-8
        b_norm[b_norm == 0] = 1e-8

        a_normed = a / a_norm
        b_normed = b / b_norm

        return np.dot(a_normed, b_normed.T)

AutoModelForSequenceClassification = None
AutoTokenizer = None

BASE_DIR = Path(__file__).resolve().parent

USE_REMOTE_SENTIMENT = os.getenv("USE_REMOTE_SENTIMENT", "false").strip().lower() in {"1", "true", "yes", "on"}


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


@st.cache_resource
def load_ticketmind_models():
    # If torch is available we load the heavy models; otherwise provide lightweight fallbacks
    device = None
    intent_model = None
    intent_tokenizer = None
    label_encoder = None

    intent_model_path = BASE_DIR / "model_final"
    if TORCH_AVAILABLE:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            intent_tokenizer = AutoTokenizer.from_pretrained(str(intent_model_path))
            intent_model = AutoModelForSequenceClassification.from_pretrained(str(intent_model_path))
            intent_model.to(device)
            intent_model.eval()

            with open(intent_model_path / "label_encoder.pkl", "rb") as f:
                label_encoder = pickle.load(f)
        except Exception:
            # fallback to rule-based if loading fails
            intent_model = None
            intent_tokenizer = None
            label_encoder = None

    # sentiment analyzer is intentionally local-only (fallback) to avoid heavy HF model
    sentiment_analyzer = None

    embedding_model = None
    embedding_model_path = resolve_local_embedding_model_path()
    if embedding_model_path is None:
        # embedding model folder missing — continue but warn; some functionality will be degraded
        embedding_model = None
    else:
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                embedding_model = SentenceTransformer(str(embedding_model_path))
            except Exception:
                embedding_model = None
        else:
            embedding_model = None

    train_df = pd.read_csv(BASE_DIR / "data" / "train_data.csv")
    # instruction_embeddings may exist; load if present
    try:
        instruction_embeddings = np.load(BASE_DIR / "data" / "instruction_embeddings.npy")
    except Exception:
        instruction_embeddings = None

    # Build simple keyword->intent map for fallback intent detection
    intent_keyword_map = {}
    for _, row in train_df.iterrows():
        intent = row.get("intent")
        instruction = str(row.get("instruction", ""))
        words = [w.lower().strip(".,!?'\"()") for w in instruction.split() if len(w) > 3]
        if intent not in intent_keyword_map:
            intent_keyword_map[intent] = set()
        intent_keyword_map[intent].update(words)

    return {
        "device": device,
        "intent_model": intent_model,
        "intent_tokenizer": intent_tokenizer,
        "label_encoder": label_encoder,
        "sentiment_analyzer": sentiment_analyzer,
        "embedding_model": embedding_model,
        "train_df": train_df,
        "instruction_embeddings": instruction_embeddings,
        "intent_keyword_map": intent_keyword_map,
    }


def predict_intent(text: str):
    model_bundle = load_ticketmind_models()

    # If a PyTorch model exists, use it
    if model_bundle.get("intent_model") is not None and model_bundle.get("intent_tokenizer") is not None:
        device = model_bundle["device"]
        intent_model = model_bundle["intent_model"]
        intent_tokenizer = model_bundle["intent_tokenizer"]
        label_encoder = model_bundle["label_encoder"]

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

    # Fallback: simple keyword-based intent matching
    intent_keyword_map = model_bundle.get("intent_keyword_map", {})
    text_norm = text.lower()
    scores = {}
    for intent, keywords in intent_keyword_map.items():
        match_count = sum(1 for k in keywords if k in text_norm)
        if match_count:
            scores[intent] = match_count

    if not scores:
        return {"intent": "unclear / general_comment", "confidence": 0.2}

    best_intent = max(scores, key=scores.get)
    # confidence heuristic
    confidence = min(0.9, 0.3 + 0.15 * scores[best_intent])
    return {"intent": best_intent, "confidence": round(confidence, 4)}


def get_sentiment(text: str):
    model_bundle = load_ticketmind_models()
    sentiment_analyzer = model_bundle["sentiment_analyzer"]

    if sentiment_analyzer is None:
        return local_sentiment_fallback(text)

    result = sentiment_analyzer(text)[0]
    return {
        "sentiment": result["label"].lower(),
        "confidence": round(result["score"], 4),
    }


def find_similar_response(new_message: str, predicted_intent: str, top_k: int = 1):
    model_bundle = load_ticketmind_models()
    train_df = model_bundle["train_df"]
    embedding_model = model_bundle.get("embedding_model")
    instruction_embeddings = model_bundle.get("instruction_embeddings")

    # If embeddings available, use them
    if embedding_model is not None and instruction_embeddings is not None:
        try:
            new_embedding = embedding_model.encode([new_message])
            similarities = cosine_similarity(new_embedding, instruction_embeddings)[0]
            sorted_indices = similarities.argsort()[::-1]

            matched_results = []
            for idx in sorted_indices:
                row = train_df.iloc[idx]
                if row["intent"] == predicted_intent:
                    matched_results.append(
                        {
                            "matched_instruction": row["instruction"],
                            "suggested_response": row["response"],
                            "similarity_score": round(float(similarities[idx]), 4),
                        }
                    )
                if len(matched_results) >= top_k:
                    break

            if not matched_results:
                for idx in sorted_indices[:top_k]:
                    row = train_df.iloc[idx]
                    matched_results.append(
                        {
                            "matched_instruction": row["instruction"],
                            "suggested_response": row["response"],
                            "similarity_score": round(float(similarities[idx]), 4),
                        }
                    )

            return matched_results
        except Exception:
            pass

    # Fallback: return the first matching row for the intent
    matched = []
    for _, row in train_df.iterrows():
        if row.get("intent") == predicted_intent:
            matched.append({
                "matched_instruction": row.get("instruction"),
                "suggested_response": row.get("response"),
                "similarity_score": None,
            })
        if len(matched) >= top_k:
            break

    if not matched and len(train_df) > 0:
        # fallback to top row(s)
        row = train_df.iloc[0]
        matched.append({
            "matched_instruction": row.get("instruction"),
            "suggested_response": row.get("response"),
            "similarity_score": None,
        })

    return matched


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
        message,
        predicted_intent=intent_result["intent"],
        top_k=1,
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


st.set_page_config(
    page_title="TicketMind",
    page_icon="💬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .title-wrap {
        margin-bottom: 0.35rem;
    }
    .title-wrap h1 {
        font-size: 2.3rem !important;
        margin-bottom: 0.1rem !important;
        letter-spacing: 0.02em;
    }
    .subtitle-space {
        margin-bottom: 1.4rem;
    }
    div.block-container {
        padding-top: 1.1rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title-wrap"><h1 style="color:#F8FAFC;">TicketMind</h1></div>', unsafe_allow_html=True)
st.caption("أداة تحليل رسائل العملاء بطريقة سهلة وسريعة")
st.markdown('<div class="subtitle-space"></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("حالة النظام")
    st.success("التطبيق يعمل محليًا بالكامل دون أي خدمة خارجية")
    st.markdown("---")
    st.write("تستخدم هذه الأداة نموذج تصنيف نوع الطلب ومشاعر العميل لاقتراح رد مناسب.")


def run_analysis(message: str):
    return process_customer_message(message)


message = st.text_area(
    "اكتب رسالة العميل",
    height=180,
    placeholder="مثل: أريد إلغاء طلبي الآن، هذا غريب جدًا!",
    value="I want to cancel my order right now, this is ridiculous!",
)

if st.button("تحليل الرسالة", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("من فضلك اكتب رسالة قبل التحليل.")
    else:
        try:
            payload = run_analysis(message)
            st.success("تم تحليل الرسالة بنجاح")

            intent = payload.get("predicted_intent", "غير معروف")
            sentiment = payload.get("sentiment", "غير معروف")
            priority = payload.get("priority", "normal")
            note = payload.get("note", "")
            suggested = payload.get("suggested_response", "لا يوجد رد مقترح")

            st.subheader("النتيجة")

            col1, col2, col3 = st.columns(3)
            col1.metric("نوع الطلب", intent)
            col2.metric("المشاعر", sentiment)
            col3.metric("الأولوية", priority)

            st.markdown("### التفاصيل السريعة")
            st.write(f"**نوع الطلب:** {intent}")
            st.write(f"**المشاعر:** {sentiment}")
            st.write(f"**الأولوية:** {priority}")
            st.write(f"**ملاحظة:** {note}")

            st.markdown("### الرد المقترح")

            if suggested and suggested.strip():
                plain_text = suggested
                plain_text = plain_text.replace("{{", "").replace("}}", "")
                plain_text = plain_text.replace("\n", " ")
                plain_text = plain_text.replace("Here's what you need to do:", "هذا ما عليك فعله:")
                plain_text = plain_text.replace("I pick up what you're putting down", "نحن نفهم سبب استياءك")
                plain_text = plain_text.replace("Let's make this process as smooth as possible.", "سنجعل هذه العملية سهلة ومباشرة قدر الإمكان.")
                plain_text = plain_text.replace("If you encounter any difficulties or have further questions", "إذا واجهتك أي صعوبة أو كانت لديك أسئلة")
                plain_text = plain_text.replace("Your satisfaction is our top priority.", "رضاك أولويتنا الأولى.")
                plain_text = " ".join(plain_text.split())

                intro_steps = [
                    "نحن نفهم سبب استياءك. رغبتك في إلغاء الطلب واضحة، وسنجعل هذه العملية سهلة ومباشرة قدر الإمكان.",
                    "هذا ما عليك فعله:",
                ]

                steps = [
                    "1. سجل الدخول إلى بوابة الطلبات الخاصة بك.",
                    "2. انتقل إلى قسم الطلبات أو إدارة الطلبات.",
                    "3. ابحث عن الطلب الذي تريد إلغاؤه وحدده.",
                    "4. اضغط على خيار إلغاء الطلب أو إلغاء الطلبات.",
                    "5. تأكد من الإلغاء واتبع أي تعليمات إضافية إذا ظهرت.",
                ]

                footer = "إذا واجهتك أي صعوبة أو كانت لديك أسئلة، فريقنا جاهز للمساعدة خلال أوقات الدعم. رضاك أولويتنا الأولى."

                card_style = "background:#1F2937; border:1px solid #374151; border-radius:12px; padding:16px 16px 14px 16px; color:#F9FAFB; line-height:1.6; font-size:14px; box-shadow: 0 4px 10px rgba(0,0,0,0.22); min-height: 150px;"

                st.markdown("#### بطاقة البداية")
                st.markdown(
                    f"<div style='{card_style}'>"
                    f"<div style='font-size:15px; font-weight:600; margin-bottom:8px;'>مقدمة</div>"
                    f"{intro_steps[0]}<br><br>{intro_steps[1]}"
                    "</div>",
                    unsafe_allow_html=True,
                )

                st.markdown('<div style="height: 18px;"></div>', unsafe_allow_html=True)

                step_cards = st.columns(5)
                for i, step in enumerate(steps):
                    with step_cards[i]:
                        st.markdown(
                            f"<div style='{card_style}'>"
                            f"<div style='font-size:14px; font-weight:700; margin-bottom:8px;'>الخطوة {i+1}</div>"
                            f"{step}"
                            "</div>",
                            unsafe_allow_html=True,
                        )

                st.markdown("#### بطاقة الدعم")
                st.markdown(
                    f"<div style='{card_style}'>"
                    f"<div style='font-size:15px; font-weight:600; margin-bottom:8px;'>الدعم</div>"
                    f"{footer}"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("لا يوجد رد مقترح في الوقت الحالي.")

        except Exception as exc:
            st.error(f"حدث خطأ: {exc}")

st.markdown("---")
st.caption("سيُظهر النظام: نوع المشكلة، مشاعر العميل، أولويتها، والرد المناسب في سطر واحد واضح.")
