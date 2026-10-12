"""Shared constants and helpers used by both the notebook and the Streamlit app."""

FEATURE_NAMES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                 "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

FEATURE_LABELS = {
    "age":      "Age (years)",
    "sex":      "Sex (1=M, 0=F)",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting BP (mmHg)",
    "chol":     "Serum Cholesterol (mg/dl)",
    "fbs":      "Fasting BS > 120 mg/dl",
    "restecg":  "Resting ECG Results",
    "thalach":  "Max Heart Rate",
    "exang":    "Exercise-Induced Angina",
    "oldpeak":  "ST Depression",
    "slope":    "Slope of Peak ST",
    "ca":       "Major Vessels (0–4)",
    "thal":     "Thalassemia Type",
}

CONTINUOUS_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak",
                   "hr_percent", "age_oldpeak"]

ACTION_MAP = {
    "age":         "Non-modifiable. Prioritise regular cardiac screening ≥ 45 yrs.",
    "sex":         "Sex-specific risk: apply gender-appropriate CVD guidelines.",
    "cp":          "Chest pain type signals angina — urgent cardiology referral.",
    "trestbps":    "Elevated BP detected. Consider antihypertensive therapy + lifestyle changes.",
    "chol":        "High cholesterol. Statin therapy and dietary intervention recommended.",
    "fbs":         "Elevated fasting glucose. Screen for diabetes, manage glycaemia.",
    "restecg":     "Abnormal ECG. Cardiology consultation and further workup indicated.",
    "thalach":     "Low max HR. Evaluate for chronotropic incompetence.",
    "exang":       "Exercise-induced angina present. Formal stress testing indicated.",
    "oldpeak":     "Significant ST depression — suggests ischaemia, further investigation needed.",
    "slope":       "Abnormal ST slope. Correlate with clinical history and imaging.",
    "ca":          "Fluoroscopy shows blocked vessels — urgent cardiology/surgical referral.",
    "thal":        "Thalassemia type noted. Combined haematological and cardiac management.",
    "hr_percent":  "Low heart-rate reserve. Consider cardiac rehabilitation programme.",
    "age_oldpeak": "High age × ST interaction — comprehensive assessment recommended.",
}


def add_features(df_in):
    """Add clinically motivated interaction features (computed on the raw scale)."""
    df_out = df_in.copy()
    df_out["hr_percent"] = df_out["thalach"] / (220 - df_out["age"])
    df_out["age_oldpeak"] = df_out["age"] * df_out["oldpeak"]
    return df_out
