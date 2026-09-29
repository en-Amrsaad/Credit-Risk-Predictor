import streamlit as st
import pandas as pd
import numpy as np
import joblib

# إعداد الصفحة
st.set_page_config(
    page_title="نظام دعم قرار الائتمان والمخاطر | Credit DSS",
    page_icon="🏦",
    layout="wide"
)

# تخصيص التصميم والخطوط في الشريط الجانبي بالـ CSS
st.markdown("""
<style>
    /* تحسين اتجاه ومظهر الشريط الجانبي */
    [data-testid="stSidebar"] {
        direction: rtl;
        text-align: right;
        background-color: #161b22;
        padding-top: 1.5rem;
    }
    
    /* عنوان وهوية الشريط الجانبي */
    .sidebar-title {
        font-size: 28px !important;
        font-weight: 800;
        color: #58a6ff;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .sidebar-subtitle {
        font-size: 14px;
        color: #8b949e;
        margin-bottom: 28px;
        border-bottom: 1px solid #30363d;
        padding-bottom: 16px;
    }

    /* تحويل خيارات التنقل إلى أزرار وبطاقات عريضة وواضحة */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        background-color: #21262d;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 14px;
        transition: all 0.25s ease-in-out;
        cursor: pointer;
        display: flex;
        align-items: center;
        width: 100%;
    }

    /* تكبير حجم النص داخل الخيارات */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label p {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #c9d1d9;
        margin-right: 12px;
    }

    /* تأثير المرور فوق الخيار */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background-color: #30363d;
        border-color: #58a6ff;
        transform: translateX(-4px);
    }

    /* تمييز الخيار النشط بلون بارز */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-checked="true"] {
        background: linear-gradient(135deg, #1f6feb 0%, #1158c7 100%);
        border-color: #58a6ff;
        box-shadow: 0 4px 14px rgba(31, 111, 235, 0.4);
    }
    
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-checked="true"] p {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ترويسة الشريط الجانبي
st.sidebar.markdown("""
<div class="sidebar-title">
    <span>🏦</span> لوحة التحكم
</div>
<div class="sidebar-subtitle">نظام اتخاذ القرارات الائتمانية الذكي</div>
""", unsafe_allow_html=True)

# قائمة التنقل المحسنة
page = st.sidebar.radio(
    "التنقل السريع:",
    [
        "📊  لوحة التحليلات والمؤشرات",
        "🎯  فحص عميل واتخاذ القرار",
        "🔮  محاكي ماذا-لو (What-If)"
    ],
    label_visibility="collapsed"
)

# تحميل النموذج والبيانات
@st.cache_resource
def load_model():
    return joblib.load('loan_model.pkl')

@st.cache_data
def load_data():
    data = pd.read_csv('credit_risk_dataset.csv')
    return data[data['person_age'] <= 100]

model = load_model()
df = load_data()

# قواميس التسميات باللغة العربية
home_ownership_labels = {
    'RENT': 'إيجار (Rent)',
    'OWN': 'ملك (Own)',
    'MORTGAGE': 'رهن عقاري (Mortgage)',
    'OTHER': 'أخرى (Other)'
}

loan_intent_labels = {
    'EDUCATION': 'تعليمي (Education)',
    'MEDICAL': 'طبي / علاجي (Medical)',
    'VENTURE': 'استثماري / مشروع ناشئ (Venture)',
    'PERSONAL': 'شخصي (Personal)',
    'HOMEIMPROVEMENT': 'تحسين وترميم منزل (Home Improvement)',
    'DEBTCONSOLIDATION': 'دمج وسداد ديون (Debt Consolidation)'
}

default_on_file_labels = {
    'N': 'لا (سجل نظيف)',
    'Y': 'نعم (يوجد تعثر سابق)'
}

grade_labels = {
    'A': 'A - ممتاز جداً',
    'B': 'B - جيد جداً',
    'C': 'C - جيد',
    'D': 'D - مقبول / حذر',
    'E': 'E - ضعيف',
    'F': 'F - حرج',
    'G': 'G - عالي الخطورة'
}

# ----------------------------------------------------
# الشاشة 1: المؤشرات والرسوم البيانية
# ----------------------------------------------------
if page == "📊  لوحة التحليلات والمؤشرات":
    st.title("📊 لوحة مؤشرات الأداء ومحفظة الائتمان")
    st.caption("تحليل وصفي واستكشافي للبيانات التاريخية لدعم قرارات الإدارة العليا")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("إجمالي طلبات المحفظة", f"{len(df):,}")
    col2.metric("متوسط القرض المطلوب", f"${df['loan_amnt'].median():,.0f}")
    col3.metric("متوسط الدخل السنوي", f"${df['person_income'].median():,.0f}")
    default_rate = (df['loan_status'].mean()) * 100
    col4.metric("معدل التعثر الإجمالي", f"{default_rate:.1f}%")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("توزيع القروض حسب الغرض")
        intent_counts = df['loan_intent'].map(loan_intent_labels).value_counts()
        st.bar_chart(intent_counts)

    with c2:
        st.subheader("نسبة التعثر حسب التصنيف الائتماني (%)")
        grade_defaults = df.groupby('loan_grade')['loan_status'].mean() * 100
        st.line_chart(grade_defaults)

    st.subheader("سجلات عشوائية من قاعدة البيانات")
    st.dataframe(df.sample(7), use_container_width=True)

# ----------------------------------------------------
# الشاشة 2: فحص العميل وقرار الائتمان
# ----------------------------------------------------
elif page == "🎯  فحص عميل واتخاذ القرار":
    st.title("🎯 محرك اتخاذ القرار الائتماني")
    st.caption("أدخل بيانات العميل للحصول على تقييم شامل للمخاطر وتوصية النظام التلقائية")

    with st.form("loan_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 البيانات الشخصية")
            person_age = st.number_input("عمر العميل", 18, 80, 29)
            person_income = st.number_input("الدخل السنوي ($)", 2000, 500000, 45000, step=1000)
            person_emp_length = st.number_input("سنوات الخدمة / العمل", 0, 45, 3)
            person_home_ownership = st.selectbox(
                "ملكية السكن",
                options=list(home_ownership_labels.keys()),
                format_func=lambda x: home_ownership_labels[x]
            )

        with col2:
            st.markdown("##### 💰 تفاصيل التمويل")
            loan_amnt = st.number_input("المبلغ المطلوب ($)", 500, 50000, 12000, step=500)
            loan_int_rate = st.number_input("نسبة الفائدة المقترحة (%)", 1.0, 30.0, 10.5, step=0.1)
            loan_intent = st.selectbox(
                "الغرض من التمويل",
                options=list(loan_intent_labels.keys()),
                format_func=lambda x: loan_intent_labels[x]
            )

        with col3:
            st.markdown("##### 📈 التاريخ الائتماني")
            loan_grade = st.selectbox(
                "التصنيف الائتماني",
                options=list(grade_labels.keys()),
                format_func=lambda x: grade_labels[x]
            )
            cb_person_default_on_file = st.selectbox(
                "هل يوجد سجل تعثر سابق؟",
                options=list(default_on_file_labels.keys()),
                format_func=lambda x: default_on_file_labels[x]
            )
            cb_person_cred_hist_length = st.number_input("طول السجل الائتماني (سنوات)", 1, 30, 4)

        submit = st.form_submit_button("🔍 تقييم الجدارة وإصدار القرار", use_container_width=True)

    if submit:
        loan_ratio = loan_amnt / person_income if person_income > 0 else 0

        sample = pd.DataFrame([{
            'person_age': person_age,
            'person_income': person_income,
            'person_emp_length': person_emp_length,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_ratio,
            'cb_person_cred_hist_length': cb_person_cred_hist_length,
            'person_home_ownership': person_home_ownership,
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'cb_person_default_on_file': cb_person_default_on_file
        }])

        prob = model.predict_proba(sample)[0][1] * 100

        st.markdown("---")
        st.subheader("📋 نتيجة التقييم وتوصية النظام")

        r1, r2 = st.columns([1, 2])
        with r1:
            st.metric("مؤشر مخاطرة التعثر", f"{prob:.1f}%")
            st.progress(int(prob))
            st.write(f"**نسبة القسط/القرض للدخل:** `{loan_ratio * 100:.1f}%`")

        with r2:
            if prob < 25:
                st.success("### ✅ القرار: قبول مباشر (مخاطر منخفضة)")
                st.write("**التوصية:** الجدارة الائتمانية ممتازة. يُنصح بالموافقة الفورية على منح التمويل.")
            elif prob < 55:
                st.warning("### ⚠️ القرار: مراجعة ائتمانية بشروط إضافية (مخاطر متوسطة)")
                st.write("**التوصية:** المعاملة مقبولة بشرط تقديم ضامن كفيل أو خفض مبلغ التمويل بنسبة 20%.")
            else:
                st.error("### ❌ القرار: رفض الطلب (مخاطر مرتفعة)")
                st.write("**التوصية:** تجاوز حدود المخاطر الائتمانية المعتمدة. يُوصى بالاعتذار عن قبول الطلب.")

# ----------------------------------------------------
# الشاشة 3: محاكاة السيناريوهات
# ----------------------------------------------------
elif page == "🔮  محاكي ماذا-لو (What-If)":
    st.title("🔮 محاكي القرارات والسيناريوهات البديلة (What-If)")
    st.caption("أداة لدراسة أثر تعديل شروط التمويل لتحويل النتيجة من رفض إلى قبول")

    col_ctrl, col_view = st.columns([1, 1])

    with col_ctrl:
        sim_income = st.slider("الدخل السنوي المفترض للعميل ($)", 10000, 120000, 35000, step=2500)
        sim_amount = st.slider("تعديل مبلغ القرض المطلوب ($)", 1000, 40000, 15000, step=1000)
        sim_rate = st.slider("تعديل معدل الفائدة (%)", 5.0, 24.0, 11.0, step=0.5)

    with col_view:
        sim_sample = pd.DataFrame([{
            'person_age': 28,
            'person_income': sim_income,
            'person_emp_length': 3.0,
            'loan_amnt': sim_amount,
            'loan_int_rate': sim_rate,
            'loan_percent_income': sim_amount / sim_income,
            'cb_person_cred_hist_length': 3,
            'person_home_ownership': 'RENT',
            'loan_intent': 'PERSONAL',
            'loan_grade': 'B',
            'cb_person_default_on_file': 'N'
        }])

        sim_risk = model.predict_proba(sim_sample)[0][1] * 100

        st.subheader("المخاطرة المتوقعة للسيناريو")
        st.metric("نسبة المخاطرة", f"{sim_risk:.1f}%")
        st.progress(int(sim_risk))

        if sim_risk < 25:
            st.success("السيناريو الحالي يحقق شروط القبول المباشر.")
        elif sim_risk < 55:
            st.warning("السيناريو في منطقة المراجعة والضمانات الإضافية.")
        else:
            st.error("السيناريو ما زال في منطقة الخطر المرتفع.")