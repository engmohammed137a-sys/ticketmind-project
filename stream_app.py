import os

import requests
import streamlit as st

try:
    from app import process_customer_message
except Exception:  # pragma: no cover
    process_customer_message = None

API_URL = os.getenv("API_URL")

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
        st.success("التطبيق يعمل محليًا بدون API خارجي")

    st.markdown("---")
    st.write("تستخدم هذه الأداة نموذج تصنيف نوع الطلب ومشاعر العميل لاقتراح رد مناسب.")


def run_analysis(message: str):
    if API_URL:
        response = requests.post(API_URL, json={"message": message}, timeout=120)
        payload = response.json()
        if response.status_code != 200:
            raise RuntimeError(payload.get("error", "حدث خطأ غير متوقع."))
        return payload

    if process_customer_message is None:
        raise RuntimeError("لم يتم تحميل منطق التحليل. تأكد من وجود app.py")

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
            intent_conf = payload.get("intent_confidence", 0)
            sentiment_conf = payload.get("sentiment_confidence", 0)
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
