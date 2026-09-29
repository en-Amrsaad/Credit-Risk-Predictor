import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. قراءة البيانات
data_path = 'credit_risk_dataset.csv'
df = pd.read_csv(data_path)

print("حجم البيانات الأصلي:", df.shape)

# 2. تنظيف القيم الشاذة الواضحة (Outliers)
# استبعاد السجلات غير المنطقية (عمر أكبر من 100 أو سنوات عمل أكثر من 60)
df = df[df['person_age'] <= 100]
df = df[df['person_emp_length'] <= 60]

# 3. تحديد المتغيرات المستقلة والهدف (Target)
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# 4. تحديد الأعمدة الرقمية والنصية
numeric_features = [
    'person_age', 'person_income', 'person_emp_length', 
    'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
    'cb_person_cred_hist_length'
]

categorical_features = [
    'person_home_ownership', 'loan_intent', 
    'loan_grade', 'cb_person_default_on_file'
]

# 5. بناء معالجات البيانات (Pipelines)
# للأرقام: تعويض القيم المفقودة بالمتوسط
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# للنصوص: تعويض المفقود بالوضع الأكثر تكراراً ثم تحويله إلى One-Hot
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 6. دمج المعالجة مع النموذج
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 7. تقسيم البيانات للتدريب والاختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("جاري تدريب المودل...")
full_pipeline.fit(X_train, y_train)

# 8. تقييم النتائج
y_pred = full_pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n--- تقرير دقة النموذج ---")
print(f"Accuracy: {acc * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

# 9. حفظ النموذج في ملف جاهز للاستخدام في الـ Dashboard
joblib.dump(full_pipeline, 'loan_model.pkl')
print("تم حفظ المودل بنجاح باسم: loan_model.pkl")