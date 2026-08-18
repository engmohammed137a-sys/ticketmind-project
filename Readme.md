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

### 3. تشغيل واجهة Streamlit

```bash
streamlit run stream_app.py
```

### 4. إذا كنت تريد استخدام API خارجي

ضع متغير البيئة `API_URL` في بيئة التشغيل، مثال:

```bash
set API_URL=https://your-api-domain.com/analyze
```

إذا لم يتم تعيين المتغير، سيعمل التطبيق مباشرة من المنطق المحلي داخل [app.py](app.py) دون الحاجة إلى Flask أو localhost.

## نقاط النهاية (لـ API منفصل)

### POST /analyze

```bash
curl -X POST https://your-api-domain.com/analyze \
  -H "Content-Type: application/json" \
  -d '{"message":"I want to cancel my order right now, this is ridiculous!"}'
```

### GET /health

```bash
curl https://your-api-domain.com/health
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
- المشروع جاهز للاستخدام على Streamlit Cloud طالما تم تعيين `API_URL` عند الحاجة، أو ترك القيمة فارغة لاستخدام المنطق المحلي.
- لا تستخدم `localhost` في النشر لأن هذا لا يعمل على خدمات الاستضافة مثل Streamlit Cloud.

TicketMind يساعد فرق خدمة العملاء على فهم الرسالة بسرعة واختيار الرد المناسب بسهولة.
