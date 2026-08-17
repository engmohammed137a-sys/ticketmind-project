# TicketMind

TicketMind هو نظام ذكي لتحليل رسائل خدمة العملاء، ويقوم بتحديد نوع الطلب، تحليل مشاعر العميل، واقتراح رد مناسب بطريقة سريعة وسهلة.

## ماذا يفعل المشروع؟

- تصنيف نوع مشكلة العميل
- تحليل مشاعر العميل
- تحديد أولوية الرسالة
- اقتراح رد مناسب بناءً على بيانات سابقة
- عرض النتيجة من خلال واجهة مستخدم بسيطة

## هيكل المشروع

```text
api_package/
├── app.py
├── stream_app.py
├── Readme.md
├── Requirements.txt
├── model_final/
├── data/
├── model_code/
├── embedding_model_local/
└── .venv/
```

## التقنيات المستخدمة

- Flask
- Streamlit
- Transformers
- SentenceTransformers
- scikit-learn
- Pandas
- NumPy

## طريقة التشغيل

### 1. إنشاء البيئة

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. تثبيت المتطلبات

```bash
pip install -r Requirements.txt
```

### 3. تشغيل الـ API

```bash
python app.py
```

### 4. تشغيل واجهة المستخدم

```bash
streamlit run stream_app.py --server.port 8501
```

## نقاط النهاية

### POST /analyze

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"message":"I want to cancel my order right now, this is ridiculous!"}'
```

### GET /health

```bash
curl http://localhost:5000/health
```

## مثال الاستجابة

```json
{
  "predicted_intent": "cancel_order",
  "intent_confidence": 0.997,
  "sentiment": "negative",
  "sentiment_confidence": 0.9298,
  "priority": "high",
  "note": "Customer appears clearly upset - recommend reviewing the response before sending and prioritizing a quick reply",
  "suggested_response": "We understand your frustration..."
}
```

## ملاحظات

- الواجهة الحالية موجهة للمستخدم العادي وتعرض النتائج بشكل مبسط.
- يمكن تحسين المشروع لاحقًا عبر إضافة قاعدة بيانات، دعم لغات أكثر، ودعم أفضل للردود المقترحة.

TicketMind يساعد فرق خدمة العملاء على فهم الرسالة بسرعة واختيار الرد المناسب بسهولة.
