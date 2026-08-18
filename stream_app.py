import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

BASE_DIR = Path(__file__).resolve().parent
API_URL = os.getenv("API_URL")


@st.cache_resource
def load_ticketmind_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    intent_model_path = BASE_DIR / "model_final"
    intent_tokenizer = AutoTokenizer.from_pretrained(str(intent_model_path))
    intent_model = AutoModelForSequenceClassification.from_pretrained(str(intent_model_path))
    intent_model.to(device)
    intent_model.eval()

    with open(intent_model_path / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=0 if device == "cuda" else -1,
    )

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    train_df = pd.read_csv(BASE_DIR / "data" / "train_data.csv")
    instruction_embeddings = np.load(BASE_DIR / "data" / "instruction_embeddings.npy")

    return {
        "device": device,
        "intent_model": intent_model,
        "intent_tokenizer": intent_tokenizer,
        "label_encoder": label_encoder,
        "sentiment_analyzer": sentiment_analyzer,
        "embedding_model": embedding_model,
        "train_df": train_df,
        "instruction_embeddings": instruction_embeddings,
    }


def predict_intent(text: str):
    model_bundle = load_ticketmind_models()
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


def get_sentiment(text: str):
    sentiment_analyzer = load_ticketmind_models()["sentiment_analyzer"]
    result = sentiment_analyzer(text)[0]
    return {
        "sentiment": result["label"].lower(),
        "confidence": round(result["score"], 4),
    }


def find_similar_response(new_message: str, predicted_intent: str, top_k: int = 1):
    model_bundle = load_ticketmind_models()
    embedding_model = model_bundle["embedding_model"]
    train_df = model_bundle["train_df"]
    instruction_embeddings = model_bundle["instruction_embeddings"]

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
    if API_URL:
        try:
            health = requests.get(f"{API_URL.replace('/analyze', '')}/health", timeout=5)
            if health.status_code == 200:
                st.success("الخدمة الخارجية تعمل")
            else:
                st.warning("الخدمة الخارجية تستجيب")
        except Exception:
            st.warning("لا يمكن الاتصال بالخدمة الخارجية")
    else:
        st.success("التطبيق يعمل محليًا داخل نفس الملف")

    st.markdown("---")
    st.write("تستخدم هذه الأداة نموذج تصنيف نوع الطلب ومشاعر العميل لاقتراح رد مناسب.")


def run_analysis(message: str):
    if API_URL:
        response = requests.post(API_URL, json={"message": message}, timeout=120)
        payload = response.json()
        if response.status_code != 200:
            raise RuntimeError(payload.get("error", "حدث خطأ غير متوقع."))
        return payload

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
