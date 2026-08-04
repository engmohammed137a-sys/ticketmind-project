# TicketMind — نظام ذكي لتحليل رسائل خدمة العملاء

نظام يستقبل رسائل العملاء ويحللها باستخدام نماذج Transformers، ثم يحدد نوع المشكلة ومشاعر العميل، ويقترح ردًا مناسبًا لموظف خدمة العملاء، مع تحديد أولوية التذكرة تلقائيًا.

## نظرة عامة على النظام

عند استقبال رسالة عميل جديدة، يقوم النظام بـ:
1. **تصنيف نوع المشكلة** (27 نية مختلفة مثل: إلغاء طلب، استرداد أموال، مشاكل الدفع...) باستخدام نموذج DistilBERT مُدرّب خصيصًا لهذه المهمة.
2. **تحليل مشاعر العميل** (إيجابي / سلبي / محايد) باستخدام نموذج RoBERTa مُدرّب مسبقًا.
3. **البحث عن أقرب حل مشابه** من قاعدة بيانات الردود السابقة باستخدام Embeddings و Cosine Similarity.
4. **تحديد أولوية التذكرة** تلقائيًا بناءً على حالة العميل النفسية (رسائل العملاء الغاضبين تُصنَّف كأولوية عالية).
5. **اكتشاف الرسائل غير الواضحة** (مثل رسائل الشكر العامة) وتجنب اقتراح رد غير مناسب لها.

## الأداء

| المهمة | النموذج | الأداء |
|---|---|---|
| تصنيف نوع المشكلة | DistilBERT (fine-tuned) | Accuracy: 99.7% / F1: 99.7% |
| تحليل المشاعر | cardiffnlp/twitter-roberta-base-sentiment-latest | 3 فئات (positive/neutral/negative) |
| البحث عن حلول مشابهة | all-MiniLM-L6-v2 + Cosine Similarity | تشابه يصل لـ 87%+ للحالات المطابقة |

## هيكل المشروع

```
ticketmind-project/
├── app.py                          # Flask API
├── requirements.txt                # المكتبات المطلوبة
├── model_final/                    # نموذج تصنيف نوع المشكلة (مُدرّب)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── vocab.txt
│   └── label_encoder.pkl
├── data/
│   ├── train_data.csv              # بيانات التدريب النظيفة
│   ├── test_data.csv               # بيانات الاختبار
│   └── instruction_embeddings.npy  # embeddings محسوبة مسبقًا
└── notebooks/                      # سكريبتات التدريب والتحليل (Colab)
    ├── day1_data_prep.py
    ├── day2_eda.py
    └── ...
```

## البيانات

تم استخدام [Bitext Customer Support Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) من Hugging Face، ويحتوي على:
- 24,635 رسالة نظيفة بعد إزالة التكرار والقيم المفقودة
- 27 نية (Intent) مختلفة، موزعة بشكل متوازن (493–1000 مثال لكل نية)
- متوسط طول الرسالة: ~9 كلمات

## طريقة التشغيل

### 1. تثبيت المتطلبات

```bash
python -m venv venv
source venv/bin/activate   # على Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. تشغيل السيرفر

```bash
python app.py
```

السيرفر هيشتغل على `http://localhost:5000`

### 3. استخدام الـ API

**Endpoint:** `POST /analyze`

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to cancel my order right now, this is ridiculous!"}'
```

**مثال الاستجابة:**

```json
{
    "original_message": "I want to cancel my order right now, this is ridiculous!",
    "predicted_intent": "cancel_order",
    "intent_confidence": 0.997,
    "sentiment": "negative",
    "sentiment_confidence": 0.9298,
    "priority": "high",
    "note": "Customer appears clearly upset - recommend reviewing the response before sending and prioritizing a quick reply",
    "suggested_response": "I pick up what you're putting down, your need to cancel your order...",
    "similarity_score": 0.7807
}
```

**Endpoint إضافي:** `GET /health` — للتأكد من أن السيرفر يعمل.

## منهجية العمل

### 1. تصنيف نوع المشكلة
تم عمل Fine-tuning لنموذج `distilbert-base-uncased` على 27 نية باستخدام مكتبة `transformers`. تم اختيار `max_length=32` بناءً على تحليل استكشافي أظهر أن متوسط طول الرسائل لا يتجاوز 9 كلمات.

### 2. تحليل المشاعر
تم استخدام نموذج جاهز (`cardiffnlp/twitter-roberta-base-sentiment-latest`) بدلاً من التدريب من الصفر، لأنه مُدرّب أصلاً على 3 فئات (إيجابي/محايد/سلبي) وهو أنسب من نماذج SST-2 التقليدية التي تفتقر لفئة "محايد" الحقيقية.

### 3. اقتراح الرد
يتم تحويل كل رسائل بيانات التدريب إلى Embeddings باستخدام `all-MiniLM-L6-v2`، ثم عند وصول رسالة جديدة، يتم البحث عن أقرب الرسائل تشابهًا (Cosine Similarity)، مع **تفضيل الرسائل من نفس نوع المشكلة المُصنَّف** لتحسين دقة الرد المقترح.

### 4. حد الثقة (Confidence Thresholding)
إذا كانت ثقة النموذج في تصنيف نوع المشكلة أقل من 50%، يتم تصنيف الرسالة كـ "غير واضحة" بدلاً من إجبار تصنيف قد يكون خاطئًا (مثال: رسائل الشكر العامة التي لا تنتمي لأي نية محددة من الـ27 نية).

## القيود المعروفة (Limitations)

- نموذج تحليل المشاعر مُدرّب على بيانات تويتر عامة، وقد يخطئ أحيانًا في تفسير أسئلة محايدة تحتوي على كلمات ذات دلالة سلبية في سياقات أخرى.
- قاعدة بيانات الحلول المقترحة (Embeddings) مبنية على نفس التوزيع اللغوي لبيانات Bitext الصناعية، وقد تحتاج إعادة معايرة عند تطبيقها على بيانات حقيقية من نطاق عمل مختلف.
- الردود المقترحة تحتوي على متغيرات نائبة (placeholders) مثل `{{Order Number}}` تحتاج إلى تعبئة يدوية أو ربط ببيانات فعلية قبل الإرسال — النظام مصمم لمساعدة الموظف وليس للإرسال الآلي المباشر.

## التطوير المستقبلي المقترح

- واجهة بسيطة (Streamlit/Gradio) لاختبار النظام بشكل تفاعلي
- تحليل Robustness للنموذج بناءً على عمود `flags` (تأثير الأخطاء الإملائية والصياغة العامية على دقة التصنيف)
- ربط النظام بقاعدة بيانات فعلية للطلبات لتعبئة الـ placeholders تلقائيًا

## المتطلبات التقنية

Python 3.11+, راجع `requirements.txt` للمكتبات الكاملة.