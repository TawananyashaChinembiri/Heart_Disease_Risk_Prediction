import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from heart_utils import ACTION_MAP, FEATURE_LABELS, FEATURE_NAMES, add_features

ROOT = Path(__file__).parent
MODELS, FIGURES, REPORTS, ASSETS = (ROOT / d for d in ("models", "figures", "reports", "assets"))

st.set_page_config(page_title="Heart Disease Risk Prediction", page_icon=str(ASSETS / "favicon.png"), layout="wide")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
@property --num {syntax: '<integer>'; initial-value: 0; inherits: false;}
:root {--ink:#0f172a; --muted:#64748b; --red:#d6244f; --navy:#0b1b3a; --line:#e5e9f2; --card:#ffffff;}
html, body, [class*="css"], .stApp {font-family:'Inter', system-ui, sans-serif; color:var(--ink);}
.stApp {background: radial-gradient(1200px 500px at 85% -10%, #ffe4ea 0%, transparent 60%), #f6f8fc;}
#MainMenu, footer, header[data-testid="stHeader"] {visibility:hidden; height:0;}
.block-container {padding-top:1.4rem; max-width:1250px;}
@keyframes rise {from {opacity:0; transform:translateY(18px);} to {opacity:1; transform:none;}}
@keyframes beat {0%,100% {transform:scale(1);} 14% {transform:scale(1.10);} 28% {transform:scale(1);} 42% {transform:scale(1.07);} 70% {transform:scale(1);}}
@keyframes trace {from {stroke-dashoffset:1;} to {stroke-dashoffset:-1;}}
@keyframes count {to {--num: var(--to);}}
@keyframes ring {from {stroke-dashoffset:339.3;} to {stroke-dashoffset:var(--off);}}
@keyframes drift {0%,100% {transform:translateY(0);} 50% {transform:translateY(-10px);}}
@keyframes shine {from {background-position:0% 50%;} to {background-position:200% 50%;}}
.rise {animation: rise .7s cubic-bezier(.2,.7,.2,1) both;}
.hero {position:relative; overflow:hidden; border-radius:24px; padding:2.6rem 2.8rem; color:#fff;
 background: linear-gradient(120deg, #0b1b3a 0%, #17306b 55%, #7b1237 100%); box-shadow:0 24px 60px -24px rgba(11,27,58,.7);}
.hero::before {content:''; position:absolute; inset:0; background-image: radial-gradient(rgba(255,255,255,.08) 1px, transparent 1px); background-size:22px 22px;}
.hero-grid {position:relative; z-index:1; display:grid; grid-template-columns: 1fr 190px; gap:2rem; align-items:center;}
.eyebrow {letter-spacing:.18em; text-transform:uppercase; font-size:.72rem; color:#ffb3c1; font-weight:600;}
.hero h1 {font-family:'Playfair Display', serif; font-size:2.9rem; line-height:1.1; margin:.5rem 0 .8rem; color:#fff; padding:0;}
.hero p {color:#cbd5e8; max-width:640px; font-size:1.02rem; margin:0;}
.hero-heart {width:170px; height:170px; justify-self:center; filter:drop-shadow(0 12px 28px rgba(255,80,110,.55)); animation: beat 1.5s ease-in-out infinite, drift 5s ease-in-out infinite;}
.ecg {position:absolute; left:0; right:0; bottom:10px; width:100%; height:64px; opacity:.35; z-index:0;}
.ecg-line {stroke-dasharray:1; stroke-dashoffset:1; animation: trace 4.5s linear infinite;}
.chips {display:flex; gap:.8rem; flex-wrap:wrap; margin-top:1.5rem; position:relative;}
.chip {background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.2); backdrop-filter:blur(6px); border-radius:14px; padding:.6rem 1.1rem; min-width:118px;}
.chip b {display:block; font-size:1.55rem; font-weight:700; animation: count 1.8s ease-out both; counter-reset:num var(--num);}
.chip b::after {content:counter(num);}
.chip span {font-size:.74rem; color:#cbd5e8; letter-spacing:.04em;}
.card {background:var(--card); border:1px solid var(--line); border-radius:18px; padding:1.3rem 1.5rem; box-shadow:0 10px 30px -18px rgba(15,23,42,.35); transition: transform .25s ease, box-shadow .25s ease;}
.card:hover {transform:translateY(-3px); box-shadow:0 18px 40px -18px rgba(15,23,42,.45);}
.card h4 {margin:0 0 .35rem; font-size:1.02rem; font-weight:600;}
.card p {margin:0; color:var(--muted); font-size:.9rem; line-height:1.5;}
.section {font-family:'Playfair Display', serif; font-size:1.55rem; margin:1.6rem 0 .8rem; padding:0;}
.section small {display:block; font-family:'Inter'; font-size:.85rem; color:var(--muted); font-weight:400; margin-top:.2rem;}
.gauge-wrap {text-align:center;}
.gauge {width:230px; height:230px;}
.gauge .track {stroke:#eef1f7;}
.gauge .bar {stroke-dasharray:339.3; stroke-dashoffset:339.3; animation: ring 1.6s cubic-bezier(.3,.7,.2,1) .15s forwards; stroke-linecap:round;}
.gwrap {position:relative; width:230px; height:230px; margin:0 auto;}
.gcenter {position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;}
.gauge-num {font-size:3rem; font-weight:700; line-height:1; animation: count 1.6s cubic-bezier(.3,.7,.2,1) .15s both; counter-reset:num var(--num);}
.gauge-num::after {content:counter(num) "%";}
.gauge-cap {color:var(--muted); font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; margin-top:.35rem;}
.badge {display:inline-block; padding:.4rem 1.1rem; border-radius:999px; font-weight:600; font-size:.88rem; margin-top:.9rem; color:#fff;}
.badge.high {background:linear-gradient(90deg,#d6244f,#ff6b81,#d6244f); background-size:200% 100%; animation: shine 3s linear infinite;}
.badge.mid {background:#e0a100;} .badge.low {background:#15935f;}
.rec {display:flex; gap:1rem; align-items:flex-start; background:#fff; border:1px solid var(--line); border-left:4px solid var(--red); border-radius:14px; padding:.9rem 1.1rem; margin-bottom:.7rem; animation: rise .6s cubic-bezier(.2,.7,.2,1) both; transition: transform .2s ease;}
.rec:hover {transform:translateX(5px);}
.rec .ico {flex:none; width:36px; height:36px; border-radius:10px; background:#ffe9ee; display:grid; place-items:center;}
.rec b {font-size:.95rem;} .rec .sv {color:var(--red); font-weight:600; font-size:.8rem; margin-left:.4rem;}
.rec p {margin:.15rem 0 0; color:var(--muted); font-size:.88rem;}
.bar-bg {height:5px; background:#f1f3f9; border-radius:5px; margin-top:.5rem; overflow:hidden;}
.bar-fg {height:100%; border-radius:5px; background:linear-gradient(90deg,#ff6b81,#d6244f); transform-origin:left; animation: grow 1.2s ease-out both;}
@keyframes grow {from {transform:scaleX(0);} to {transform:scaleX(1);}}
.step {position:relative; padding-left:3.2rem; margin-bottom:1rem;}
.step .n {position:absolute; left:0; top:0; width:2.3rem; height:2.3rem; border-radius:50%; background:linear-gradient(135deg,#17306b,#d6244f); color:#fff; display:grid; place-items:center; font-weight:700;}
.step::before {content:''; position:absolute; left:1.1rem; top:2.4rem; bottom:-1.1rem; width:2px; background:linear-gradient(#d6244f55,transparent);}
.step:last-child::before {display:none;}
.notice {background:#fff7e6; border:1px solid #f5d38a; color:#7a5200; border-radius:12px; padding:.7rem 1rem; font-size:.85rem;}
.footer {text-align:center; color:var(--muted); font-size:.8rem; margin:2.5rem 0 .5rem;}
section[data-testid="stSidebar"] {background:linear-gradient(180deg,#ffffff,#fdeff3); border-right:1px solid var(--line);}
.side-brand {display:flex; align-items:center; gap:.7rem; margin-bottom:.3rem;}
.side-brand svg {width:38px; height:38px; animation: beat 1.5s ease-in-out infinite;}
.side-brand b {font-family:'Playfair Display', serif; font-size:1.2rem;}
.group {font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:var(--red); font-weight:700; margin:1.1rem 0 .1rem;}
.stTabs [data-baseweb="tab-list"] {gap:.4rem; border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"] {border-radius:10px 10px 0 0; padding:.6rem 1.2rem; font-weight:600;}
.stTabs [aria-selected="true"] {color:var(--red);}
.stTabs [data-baseweb="tab-highlight"] {background:var(--red);}
div[data-testid="stImage"] img {border-radius:14px; border:1px solid var(--line); background:#fff; transition: transform .3s ease;}
div[data-testid="stImage"] img:hover {transform:scale(1.015);}
"""

ICONS = {
    "alert": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#d6244f" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l3-8 4 16 3-8h4"/></svg>',
    "data": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#17306b" stroke-width="2" stroke-linecap="round"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></svg>',
    "brain": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#17306b" stroke-width="2" stroke-linecap="round"><circle cx="6" cy="6" r="2"/><circle cx="18" cy="6" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="6" cy="18" r="2"/><circle cx="18" cy="18" r="2"/><path d="M8 6h8M7 7.5l4 3.2M17 7.5l-4 3.2M11 13.3l-4 3.4M13 13.3l4 3.4"/></svg>',
    "eye": '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#17306b" stroke-width="2" stroke-linecap="round"><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>',
}


def html(block):
    return " ".join(line.strip() for line in block.strip().splitlines())


def show(block):
    st.markdown(html(block), unsafe_allow_html=True)


def asset(name):
    return (ASSETS / name).read_text()


@st.cache_resource
def load_artifacts():
    return {
        "meta": json.loads((MODELS / "metadata.json").read_text()),
        "scaler": joblib.load(MODELS / "scaler.joblib"),
        "stacking": joblib.load(MODELS / "stacking_model.joblib"),
        "xgb": joblib.load(MODELS / "xgboost_pipeline.joblib"),
    }


CHOICES = {
    "sex": {"Female": 0, "Male": 1},
    "cp": {"Typical angina": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3},
    "fbs": {"No (up to 120 mg/dl)": 0, "Yes (above 120 mg/dl)": 1},
    "restecg": {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2},
    "exang": {"No": 0, "Yes": 1},
    "slope": {"Upsloping": 0, "Flat": 1, "Downsloping": 2},
    "thal": {"0 - Unknown/other": 0, "1 - Normal": 1, "2 - Fixed defect": 2, "3 - Reversible defect": 3},
}


def collect_inputs():
    sb, v = st.sidebar, {}
    sb.markdown(html(f'<div class="side-brand">{asset("heart.svg")}<b>CardioSense</b></div>'
                     '<div style="color:#64748b;font-size:.82rem">Enter the patient profile to estimate risk.</div>'),
                unsafe_allow_html=True)
    sb.markdown('<div class="group">Demographics</div>', unsafe_allow_html=True)
    v["age"] = sb.slider("Age (years)", 25, 80, 55)
    v["sex"] = CHOICES["sex"][sb.selectbox("Sex", list(CHOICES["sex"]), index=1)]
    sb.markdown('<div class="group">Symptoms and vitals</div>', unsafe_allow_html=True)
    v["cp"] = CHOICES["cp"][sb.selectbox("Chest pain type", list(CHOICES["cp"]))]
    v["trestbps"] = sb.slider("Resting blood pressure (mmHg)", 90, 200, 130)
    v["chol"] = sb.slider("Serum cholesterol (mg/dl)", 120, 570, 240)
    v["fbs"] = CHOICES["fbs"][sb.selectbox("Fasting blood sugar", list(CHOICES["fbs"]))]
    v["exang"] = CHOICES["exang"][sb.selectbox("Exercise-induced angina", list(CHOICES["exang"]))]
    sb.markdown('<div class="group">Test results</div>', unsafe_allow_html=True)
    v["restecg"] = CHOICES["restecg"][sb.selectbox("Resting ECG", list(CHOICES["restecg"]))]
    v["thalach"] = sb.slider("Maximum heart rate achieved", 70, 205, 150)
    v["oldpeak"] = sb.slider("ST depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
    v["slope"] = CHOICES["slope"][sb.selectbox("Slope of peak exercise ST", list(CHOICES["slope"]), index=1)]
    v["ca"] = sb.slider("Major vessels coloured by fluoroscopy", 0, 4, 0)
    v["thal"] = CHOICES["thal"][sb.selectbox("Thalassemia", list(CHOICES["thal"]), index=2)]
    return pd.DataFrame([v])[FEATURE_NAMES]


def preprocess(raw, art):
    meta = art["meta"]
    eng = add_features(raw)[meta["all_feature_names"]]
    eng[meta["continuous_cols"]] = art["scaler"].transform(eng[meta["continuous_cols"]])
    selected = meta["selected_features"]
    return eng[selected].values.astype(float), add_features(raw)[selected]


def shap_waterfall(art, x_scaled, x_display):
    clf = art["xgb"].named_steps["classifier"]
    explainer = shap.TreeExplainer(clf)
    values = explainer.shap_values(art["xgb"].named_steps["imputer"].transform(x_scaled))[0]
    names = art["meta"]["selected_features"]
    exp = shap.Explanation(values=values, base_values=float(explainer.expected_value),
                           data=x_display.values[0], feature_names=names)
    fig = plt.figure(figsize=(8, 5.5))
    shap.plots.waterfall(exp, max_display=10, show=False)
    plt.tight_layout()
    return fig, dict(zip(names, values))


def gauge(prob):
    pct = round(prob * 100)
    color, level, label = ("#15935f", "low", "Lower risk") if prob < 0.35 else \
                          ("#e0a100", "mid", "Moderate risk") if prob < 0.65 else ("#d6244f", "high", "Higher risk")
    show(f"""
    <div class="card gauge-wrap rise">
      <div class="gwrap"><svg class="gauge" viewBox="0 0 120 120">
        <circle class="track" cx="60" cy="60" r="54" fill="none" stroke-width="9"/>
        <circle class="bar" cx="60" cy="60" r="54" fill="none" stroke-width="9" stroke="{color}"
                transform="rotate(-90 60 60)" style="--off:{339.3 * (1 - prob):.1f}"/>
      </svg>
      <div class="gcenter"><div class="gauge-num" style="--to:{pct}"></div><div class="gauge-cap">probability</div></div></div>
      <span class="badge {level}">{label}</span>
      <p style="color:#64748b;font-size:.8rem;margin-top:.9rem">Stacking ensemble: RF + XGBoost + SVM with a logistic regression meta-learner.</p>
    </div>""")
    return prob >= 0.5


def risk_tab(raw, art):
    x_scaled, x_display = preprocess(raw, art)
    prob = float(art["stacking"].predict_proba(x_scaled)[0, 1])
    fig, contrib = shap_waterfall(art, x_scaled, x_display)

    c1, c2 = st.columns([1, 1.9], gap="large")
    with c1:
        gauge(prob)
    with c2:
        show('<div class="section" style="margin-top:0">Why this prediction?<small>SHAP waterfall on the XGBoost component. '
             'Red pushes towards disease, blue away from it (log-odds).</small></div>')
        st.pyplot(fig, clear_figure=True)

    show('<div class="section">Actionable explanation<small>Recommendations for each factor that raises this patient\'s predicted risk.</small></div>')
    drivers = sorted(((f, v) for f, v in contrib.items() if v >= 0.03), key=lambda t: -t[1])
    top = drivers[0][1] if drivers else 1
    if drivers:
        for i, (feat, val) in enumerate(drivers):
            show(f"""
            <div class="rec" style="animation-delay:{0.08 * i:.2f}s">
              <div class="ico">{ICONS['alert']}</div>
              <div style="flex:1"><b>{FEATURE_LABELS.get(feat, feat)}</b><span class="sv">SHAP {val:+.2f}</span>
              <p>{ACTION_MAP.get(feat, 'Monitor closely.')}</p>
              <div class="bar-bg"><div class="bar-fg" style="width:{100 * val / top:.0f}%;animation-delay:{0.08 * i + .2:.2f}s"></div></div></div>
            </div>""")
    else:
        show('<div class="card"><p>No factor materially increases this patient\'s predicted risk.</p></div>')
    show('<div class="notice">These outputs reflect patterns learned from this dataset, whose feature coding differs from the textbook '
         'UCI description (for example, fewer coloured vessels raises predicted risk here). Interpret with care.</div>')


def performance_tab():
    metrics_path = REPORTS / "model_metrics.csv"
    show('<div class="section" style="margin-top:.4rem">Held-out test-set performance<small>20% stratified split, '
         'results never seen during training or tuning.</small></div>')
    if metrics_path.exists():
        df = pd.read_csv(metrics_path).set_index("Model")
        best = df["AUC-ROC"].idxmax()
        cols = st.columns(4)
        kpis = [("Best AUC-ROC", f"{df['AUC-ROC'].max():.3f}", best),
                ("Best accuracy", f"{df['Accuracy'].max():.3f}", df["Accuracy"].idxmax()),
                ("Best recall", f"{df['Recall'].max():.3f}", df["Recall"].idxmax()),
                ("Best MCC", f"{df['MCC'].max():.3f}", df["MCC"].idxmax())]
        for i, (col, (name, val, who)) in enumerate(zip(cols, kpis)):
            col.markdown(html(f'<div class="card rise" style="animation-delay:{i * .1}s"><p>{name}</p>'
                              f'<h4 style="font-size:1.8rem;margin:.2rem 0">{val}</h4><p>{who}</p></div>'),
                         unsafe_allow_html=True)
        st.write("")
        st.dataframe(df.style.format("{:.3f}").highlight_max(color="#fbd5dd"), width="stretch")
    for title, files in [("ROC and precision-recall curves", ["ROC_curves_all_models.png", "precision_recall_curves_all_models.png"]),
                         ("Global explainability", ["SHAP_summary_beeswarm_plot.png", "SHAP_global_feature_importance_bar_plot.png"]),
                         ("Local explanations (LIME)", ["LIME_local_explanation_high_risk.png", "LIME_local_explanation_low_risk.png"])]:
        show(f'<div class="section">{title}</div>')
        for col, f in zip(st.columns(2, gap="large"), files):
            if (FIGURES / f).exists():
                col.image(str(FIGURES / f))


def about_tab():
    show('<div class="section" style="margin-top:.4rem">Three ideas behind the project</div>')
    cards = [("data", "Clean, honest data", "1,025 rows reduced to 302 unique patients after removing 723 duplicates, so evaluation is not inflated by repeats."),
             ("brain", "Stacked ensemble", "Random forest, XGBoost and SVM are tuned and combined by a logistic regression meta-learner."),
             ("eye", "Explainable by design", "SHAP and LIME show why each prediction was made, and SHAP values drive the recommendations.")]
    for i, (col, (ic, title, text)) in enumerate(zip(st.columns(3, gap="large"), cards)):
        col.markdown(html(f'<div class="card rise" style="animation-delay:{i * .12}s;height:100%">'
                          f'<div style="margin-bottom:.6rem">{ICONS[ic]}</div><h4>{title}</h4><p>{text}</p></div>'),
                     unsafe_allow_html=True)
    show('<div class="section">Pipeline</div>')
    steps = [("Preprocess", "Remove duplicates, check types and missing values."),
             ("Engineer features", "Heart-rate reserve and age x ST-depression interaction."),
             ("Balance and scale", "SMOTE on the training split only, then standard scaling."),
             ("Select and tune", "RFECV feature selection, grid-search for four base models."),
             ("Stack and evaluate", "Meta-learner on top, scored on the untouched test set."),
             ("Explain", "SHAP, LIME and clinical action mapping.")]
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        for i, (t, d) in enumerate(steps):
            show(f'<div class="step rise" style="animation-delay:{i * .1}s"><div class="n">{i + 1}</div><b>{t}</b>'
                 f'<p style="margin:.1rem 0 0;color:#64748b;font-size:.9rem">{d}</p></div>')
    with right:
        img = FIGURES / "feature_importance_bar_chart.png"
        if img.exists():
            st.image(str(img), caption="Random forest feature importance")
        img = FIGURES / "class_distribution.png"
        if img.exists():
            st.image(str(img), caption="Class distribution after de-duplication")
    show('<div class="footer">Tawananyasha Nigel Chinembiri, University of Zimbabwe, Department of Analytics and Informatics. '
         'Supervisor: Mr. Zimowa.</div>')


def hero(art):
    meta = art["meta"]
    chips = [(302, "unique patients"), (5, "models compared"), (len(meta["selected_features"]), "features used"),
             (meta["n_test"], "test patients")]
    chip_html = "".join(f'<div class="chip"><b style="--to:{n}"></b><span>{lbl}</span></div>' for n, lbl in chips)
    show(f"""
    <div class="hero rise">
      <div class="hero-grid">
        <div>
          <div class="eyebrow">Machine learning and explainable AI</div>
          <h1>Heart Disease Risk Prediction</h1>
          <p>Estimate the likelihood of heart disease from clinical measurements, and see exactly which factors
             drive each prediction and what to do about them.</p>
          <div class="chips">{chip_html}</div>
        </div>
        {asset('heart.svg')}
      </div>
      {asset('ecg.svg')}
    </div>""")


def main():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    if not (MODELS / "stacking_model.joblib").exists():
        st.error("Model artefacts not found. Run heart_disease_prediction.ipynb first to create the models folder.")
        st.stop()
    art = load_artifacts()
    raw = collect_inputs()
    hero(art)
    st.write("")
    t1, t2, t3 = st.tabs(["Risk assessment", "Model performance", "About the project"])
    with t1:
        risk_tab(raw, art)
    with t2:
        performance_tab()
    with t3:
        about_tab()


main()
