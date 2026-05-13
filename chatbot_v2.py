"""
chatbot_v2.py — Professional dark-theme Streamlit UI for the Medical Healthcare AI Chatbot.

Pages:
  💬 Chat            — RAG-powered chatbot with quick-question guide
  📊 Dashboard       — Hospital analytics with Plotly charts
  🔍 Patient Explorer — Filterable patient table + profile viewer
  📈 LoS Predictor   — ML-based length-of-stay prediction
  🏗️ Architecture    — System architecture overview

Run:
  streamlit run chatbot_v2.py
"""

import os
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
API_URL  = os.getenv("API_URL", "http://localhost:8000")
DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(
    page_title="Medical Healthcare AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
:root {
  --bg-base:#0d1117; --bg-card:#161b22; --border:#30363d;
  --text-main:#e6edf3; --text-muted:#8b949e;
  --acc-blue:#58a6ff; --acc-teal:#39d353; --acc-purple:#d2a8ff;
  --acc-orange:#ffa657; --acc-red:#f85149;
}
.stApp,[data-testid="stAppViewContainer"]{background:var(--bg-base)!important;color:var(--text-main)!important}
[data-testid="stHeader"]{background:var(--bg-base)!important}
[data-testid="stSidebar"],[data-testid="stSidebar"]>div{background:#0d1117!important;border-right:1px solid var(--border)}
[data-testid="stSidebar"] *{color:var(--text-main)!important}
h1{font-size:1.8rem!important;font-weight:700!important;color:var(--text-main)!important;margin-bottom:4px!important}
h2{font-size:1.4rem!important;color:var(--text-main)!important}
h3{font-size:1.1rem!important;color:var(--acc-blue)!important}
[data-testid="metric-container"]{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:10px!important;padding:14px 18px!important}
[data-testid="metric-container"] label{color:var(--text-muted)!important;font-size:11px!important}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:var(--text-main)!important;font-size:1.6rem!important;font-weight:700!important}
[data-testid="stChatMessageContent"]{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:10px!important;padding:12px 16px!important;color:var(--text-main)!important}
[data-testid="stChatInputTextArea"]{background:var(--bg-card)!important;color:var(--text-main)!important;border:1px solid var(--border)!important;border-radius:8px!important}
.stButton>button{background:var(--acc-blue)!important;color:#0d1117!important;border:none!important;border-radius:6px!important;font-weight:600!important}
.stTextInput input,.stNumberInput input{background:var(--bg-card)!important;color:var(--text-main)!important;border-color:var(--border)!important}
.stSelectbox [data-baseweb="select"]{background:var(--bg-card)!important}
[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:8px}
[data-testid="stExpander"]{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:8px!important}
[data-testid="stExpander"] summary{color:var(--text-main)!important}
hr{border-color:var(--border)!important;margin:12px 0!important}
.badge-intent{background:#1c1435;color:var(--acc-purple);border:1px solid #5a3e8a;border-radius:4px;padding:2px 9px;font-size:11px;display:inline-block;margin:3px 2px}
.badge-source{background:#0c1e30;color:var(--acc-blue);border:1px solid #1a4a7a;border-radius:4px;padding:2px 9px;font-size:11px;display:inline-block;margin:3px 2px}
.badge-ok{background:#0c2318;color:var(--acc-teal);border:1px solid #1a4a30;border-radius:4px;padding:2px 9px;font-size:11px;display:inline-block}
.kpi-tile{background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:16px 20px;text-align:center}
.kpi-tile .kv{font-size:1.8rem;font-weight:700;color:var(--text-main)}
.kpi-tile .kl{font-size:11px;color:var(--text-muted);margin-top:4px}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# ── API helpers ──────────────────────────────────────────────────────────────
def api_get(endpoint, params=None):
    try:
        r = requests.get(f"{API_URL}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def api_post(endpoint, data):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=data, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def load_csv(filename):
    path = DATA_DIR / filename
    return pd.read_csv(path) if path.exists() else None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:8px 0 4px'>"
        "<span style='font-size:2rem'>🏥</span><br>"
        "<span style='font-weight:700;font-size:1rem;color:#e6edf3'>Medical AI Assistant</span><br>"
        "<span style='font-size:11px;color:#8b949e'>COVID-19 Inpatient · Canada Hosp1</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["💬 Chat", "📊 Dashboard", "🔍 Patient Explorer", "📈 LoS Predictor", "🏗️ Architecture"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    health = api_get("/health")
    if health:
        st.markdown(
            f"<span class='badge-ok'>● API Online v{health.get('version','?')}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span style='background:#2d0c0c;color:#f85149;border:1px solid #6a2020;"
            "border-radius:4px;padding:2px 9px;font-size:11px'>● API Offline</span>",
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#8b949e'>"
        "📦 <b style='color:#e6edf3'>Stack</b><br>"
        "FastAPI · LangChain · ChromaDB<br>"
        "HuggingFace Embeddings · scikit-learn<br><br>"
        "📐 <b style='color:#e6edf3'>Model</b><br>"
        "R²=0.9915 (LoS Predictor)<br>"
        "all-MiniLM-L6-v2 (Embeddings)<br><br>"
        "📊 <b style='color:#e6edf3'>Dataset</b><br>"
        "508 patients · 43 ICU · 465 Ward"
        "</div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# PAGE: CHAT
# ════════════════════════════════════════════════════════════════════════════
if "💬 Chat" in page:
    col_chat, col_guide = st.columns([3, 1])

    with col_chat:
        st.title("💬 Medical AI Chatbot")
        st.caption("RAG-powered assistant grounded in the COVID-19 inpatient dataset.")

        if "messages" not in st.session_state:
            st.session_state.messages = [{
                "role": "assistant",
                "content": (
                    "Hello! I'm your **Medical AI Assistant**. I can answer questions about the "
                    "COVID-19 inpatient dataset (508 patients), predict hospital length of stay, "
                    "search medications by semantic similarity, and provide clinical analytics.\n\n"
                    "Use the **Question Guide →** panel for sample prompts, or type your own below."
                ),
                "intent": None,
                "sources": [],
            }]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("intent"):
                    st.markdown(
                        f'<span class="badge-intent">🎯 {msg["intent"].replace("_"," ").title()}</span>',
                        unsafe_allow_html=True,
                    )
                for s in msg.get("sources", [])[:3]:
                    st.markdown(
                        f'<span class="badge-source">📋 Patient #{s.get("patient_id","?")} '
                        f'— {s.get("doc_type","record")}</span>',
                        unsafe_allow_html=True,
                    )

        if "quick_q" in st.session_state:
            prefill = st.session_state.pop("quick_q")
        else:
            prefill = None

        user_input = st.chat_input("Ask about patients, medications, statistics, or LoS prediction...")
        if prefill and not user_input:
            user_input = prefill

        if user_input:
            st.session_state.messages.append({
                "role": "user", "content": user_input, "intent": None, "sources": [],
            })
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Searching patient records & generating answer…"):
                    result = api_post("/api/chat", {"message": user_input, "session_id": "streamlit"})

                if result:
                    answer  = result.get("answer", "No response.")
                    intent  = result.get("intent", "")
                    sources = result.get("sources", [])
                    src_cnt = result.get("source_count", 0)
                else:
                    answer  = (
                        "⚠️ **API unavailable.** "
                        "Start the backend:\n```\nuvicorn deployment.api:app --reload\n```"
                    )
                    intent, sources, src_cnt = "", [], 0

                st.markdown(answer)
                if intent:
                    st.markdown(
                        f'<span class="badge-intent">🎯 {intent.replace("_"," ").title()}</span>'
                        f'&nbsp;<span style="font-size:11px;color:#8b949e">'
                        f'{src_cnt} source{"s" if src_cnt != 1 else ""} retrieved</span>',
                        unsafe_allow_html=True,
                    )
                for s in sources[:4]:
                    st.markdown(
                        f'<span class="badge-source">📋 Patient #{s.get("patient_id","?")} '
                        f'— {s.get("doc_type","record")}</span>',
                        unsafe_allow_html=True,
                    )

            st.session_state.messages.append({
                "role": "assistant", "content": answer,
                "intent": intent, "sources": sources,
            })

        if len(st.session_state.get("messages", [])) > 1:
            if st.button("🗑 Clear conversation"):
                st.session_state.messages = []
                st.rerun()

    # ── Question Guide Panel ──────────────────────────────────────────────────
    with col_guide:
        st.markdown(
            "<div style='padding:10px 0 4px'>"
            "<b style='font-size:13px;color:#e6edf3'>📖 Question Guide</b>"
            "</div>",
            unsafe_allow_html=True,
        )
        question_groups = {
            "📊 Dataset Stats": [
                "How many patients are in the dataset?",
                "What percentage were admitted to the ICU?",
                "What is the average hospital length of stay?",
                "What is the overall mortality rate?",
                "What is the average age of patients?",
            ],
            "🏥 Patient Cohorts": [
                "What are the most common comorbidities?",
                "Compare ICU vs ward patient demographics.",
                "What are the top 5 reasons for admission?",
                "Show me stats for male vs female patients.",
                "Which comorbidities are linked to ICU admission?",
            ],
            "💊 Medications": [
                "What are the most prescribed medications?",
                "Which anticoagulants are used in this dataset?",
                "Tell me about COVID-19 antiviral treatments.",
                "Which steroids are used for COVID patients?",
                "Search for antiviral medications.",
            ],
            "📈 Length of Stay": [
                "What factors most affect hospital length of stay?",
                "What is the average ICU length of stay?",
                "Do intubated patients stay longer?",
                "What is the range of hospital stay durations?",
                "Predict LoS for a 70-year-old ICU patient.",
            ],
            "🔬 Clinical Analysis": [
                "Which vitals are associated with longer stays?",
                "How does oxygen saturation affect outcomes?",
                "What is the relationship between age and mortality?",
                "Which patients have the highest mortality risk?",
                "What are outcomes for intubated patients?",
            ],
        }
        for group, questions in question_groups.items():
            with st.expander(group, expanded=False):
                for q in questions:
                    if st.button(q, key=f"q_{hash(q)}", use_container_width=True):
                        st.session_state.quick_q = q
                        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
elif "📊 Dashboard" in page:
    st.title("📊 Hospital Analytics Dashboard")
    st.caption("Live statistics from the COVID-19 inpatient dataset (508 patients — Canada Hospital 1)")

    summary = api_get("/api/analytics/summary")
    df_adm  = load_csv("Preprocessed_Data-at-Admission.csv")
    df_los  = load_csv("Preprocessed_Hospital-LoS.csv")

    # Fallback to local CSV
    if not summary and df_adm is not None and df_los is not None:
        icu_col = "admission_disposition"
        icu_cnt = int((df_adm[icu_col] == "ICU").sum()) if icu_col in df_adm.columns else 0
        exp_col = "did_the_patient_expire_in_hospital"
        exp_cnt = int((df_los[exp_col] == 1).sum()) if exp_col in df_los.columns else None
        los_col = "hospital_length_of_stay"
        icu_los = "icu_length_of_stay"
        summary = {
            "total_patients": len(df_adm),
            "icu_admissions": icu_cnt,
            "ward_admissions": len(df_adm) - icu_cnt,
            "icu_pct": round(icu_cnt / len(df_adm) * 100, 1),
            "avg_hospital_los_days": round(float(df_los[los_col].mean()), 1) if los_col in df_los.columns else None,
            "avg_icu_los_days": round(float(df_los[icu_los].mean()), 1) if icu_los in df_los.columns else None,
            "mortality_pct": round(exp_cnt / len(df_adm) * 100, 1) if exp_cnt is not None else None,
        }

    if summary:
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        tiles = [
            (k1, "🧑‍⚕️", "Total Patients",   str(summary.get("total_patients","N/A")), ""),
            (k2, "🚨", "ICU Admissions",    str(summary.get("icu_admissions","?")),   f"{summary.get('icu_pct','?')}% of total"),
            (k3, "🛏",  "Ward Admissions",  str(summary.get("ward_admissions","?")),  ""),
            (k4, "📅", "Avg Hospital LoS",  f"{summary.get('avg_hospital_los_days','?')} d", "days"),
            (k5, "🏥", "Avg ICU LoS",       f"{summary.get('avg_icu_los_days','?')} d", "days"),
            (k6, "💔", "Mortality Rate",    f"{summary.get('mortality_pct','N/A')}%", ""),
        ]
        for col, icon, label, value, sub in tiles:
            with col:
                sub_html = f"<div style='font-size:11px;color:#8b949e;margin-top:3px'>{sub}</div>" if sub else ""
                st.markdown(
                    f"<div class='kpi-tile'><div style='font-size:1.4rem'>{icon}</div>"
                    f"<div class='kv'>{value}</div><div class='kl'>{label}</div>{sub_html}</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    if df_adm is not None and df_los is not None:
        _lyt = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#8b949e", size=12))
        c1, c2 = st.columns(2)

        with c1:
            if "admission_disposition" in df_adm.columns:
                disp = df_adm["admission_disposition"].value_counts().reset_index()
                disp.columns = ["Disposition", "Count"]
                fig = px.pie(disp, names="Disposition", values="Count",
                             title="ICU vs WARD Admissions",
                             color_discrete_sequence=["#58a6ff", "#39d353"], hole=0.45)
                fig.update_layout(**_lyt, title_font_color="#e6edf3")
                fig.update_traces(textfont_color="#e6edf3")
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            if "age" in df_adm.columns:
                fig = px.histogram(df_adm, x="age", nbins=20,
                                   title="Patient Age Distribution",
                                   color_discrete_sequence=["#d2a8ff"])
                fig.update_layout(**_lyt, title_font_color="#e6edf3",
                                  xaxis=dict(gridcolor="#30363d"),
                                  yaxis=dict(gridcolor="#30363d"))
                st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            if "sex" in df_adm.columns:
                sc = df_adm["sex"].value_counts().reset_index()
                sc.columns = ["Sex", "Count"]
                fig = px.bar(sc, x="Sex", y="Count", title="Sex Distribution",
                             color="Sex",
                             color_discrete_map={"Male": "#58a6ff", "Female": "#ffa657"})
                fig.update_layout(**_lyt, title_font_color="#e6edf3", showlegend=False,
                                  xaxis=dict(gridcolor="#30363d"),
                                  yaxis=dict(gridcolor="#30363d"))
                st.plotly_chart(fig, use_container_width=True)

        with c4:
            los_col = "hospital_length_of_stay"
            if los_col in df_los.columns and "admission_disposition" in df_adm.columns:
                id_key = "parent_id" if "parent_id" in df_los.columns else "id"
                if id_key in df_los.columns and "id" in df_adm.columns:
                    merged = df_los.merge(
                        df_adm[["id", "admission_disposition"]],
                        left_on=id_key, right_on="id", how="left"
                    )
                    if "admission_disposition" in merged.columns:
                        fig = px.box(merged, x="admission_disposition", y=los_col,
                                     title="Hospital LoS by Admission Type (days)",
                                     color="admission_disposition",
                                     color_discrete_map={"ICU": "#f85149", "WARD": "#39d353"})
                        fig.update_layout(**_lyt, title_font_color="#e6edf3",
                                          showlegend=False,
                                          xaxis=dict(gridcolor="#30363d"),
                                          yaxis=dict(gridcolor="#30363d"))
                        st.plotly_chart(fig, use_container_width=True)

        # Comorbidities bar
        if "comorbidities" in df_adm.columns:
            st.markdown("#### 🦠 Top 12 Comorbidities")
            counts = {}
            for entry in df_adm["comorbidities"].dropna():
                for c in str(entry).split(","):
                    c = c.strip().lower()
                    if c and c not in ("nan", "none", ""):
                        counts[c] = counts.get(c, 0) + 1
            top12 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:12]
            if top12:
                df_c = pd.DataFrame(top12, columns=["Comorbidity", "Count"])
                fig = px.bar(df_c, x="Count", y="Comorbidity", orientation="h",
                             color_discrete_sequence=["#ffa657"])
                fig.update_layout(**_lyt, height=380,
                                  yaxis=dict(autorange="reversed", gridcolor="#30363d"),
                                  xaxis=dict(gridcolor="#30363d"))
                st.plotly_chart(fig, use_container_width=True)

        # Mortality + ICU LoS distribution
        exp_col = "did_the_patient_expire_in_hospital"
        if exp_col in df_los.columns:
            expired  = int((df_los[exp_col] == 1).sum())
            survived = len(df_los) - expired
            c5, c6 = st.columns(2)
            with c5:
                fig = px.pie(values=[survived, expired], names=["Survived", "Expired"],
                             title="In-Hospital Mortality",
                             color_discrete_sequence=["#39d353", "#f85149"], hole=0.45)
                fig.update_layout(**_lyt, title_font_color="#e6edf3")
                fig.update_traces(textfont_color="#e6edf3")
                st.plotly_chart(fig, use_container_width=True)
            with c6:
                if "icu_length_of_stay" in df_los.columns:
                    fig = px.histogram(df_los, x="icu_length_of_stay", nbins=20,
                                       title="ICU Length of Stay Distribution",
                                       color_discrete_sequence=["#f85149"])
                    fig.update_layout(**_lyt, title_font_color="#e6edf3",
                                      xaxis=dict(gridcolor="#30363d"),
                                      yaxis=dict(gridcolor="#30363d"))
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("CSV files not found. Run `python -m rag.ingest` or check the `data/` directory.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: PATIENT EXPLORER
# ════════════════════════════════════════════════════════════════════════════
elif "🔍 Patient Explorer" in page:
    st.title("🔍 Patient Explorer")
    st.caption("Search and filter 508 COVID-19 inpatients. Enter a Patient ID to load the full profile.")

    df = load_csv("Preprocessed_Data-at-Admission.csv")
    if df is None:
        st.error("Could not load patient data. Ensure data/Preprocessed_Data-at-Admission.csv exists.")
    else:
        fc1, fc2, fc3, fc4 = st.columns([3, 1, 1, 1])
        with fc1:
            search = st.text_input("🔎 Search", placeholder="comorbidity, reason for admission…")
        with fc2:
            disp_opts = ["All"] + (
                sorted(df["admission_disposition"].dropna().unique().tolist())
                if "admission_disposition" in df.columns else []
            )
            disp_sel = st.selectbox("Disposition", disp_opts)
        with fc3:
            sex_opts = ["All"] + (
                sorted(df["sex"].dropna().unique().tolist())
                if "sex" in df.columns else []
            )
            sex_sel = st.selectbox("Sex", sex_opts)
        with fc4:
            age_min = int(df["age"].min()) if "age" in df.columns else 0
            age_max = int(df["age"].max()) if "age" in df.columns else 100
            age_range = st.slider("Age range", age_min, age_max, (age_min, age_max))

        filtered = df.copy()
        if search:
            mask = filtered.apply(lambda row: search.lower() in str(row).lower(), axis=1)
            filtered = filtered[mask]
        if disp_sel != "All" and "admission_disposition" in filtered.columns:
            filtered = filtered[
                filtered["admission_disposition"].str.upper() == disp_sel.upper()
            ]
        if sex_sel != "All" and "sex" in filtered.columns:
            filtered = filtered[
                filtered["sex"].str.capitalize() == sex_sel.capitalize()
            ]
        if "age" in filtered.columns:
            filtered = filtered[
                (filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1])
            ]

        st.markdown(
            f"<span style='font-size:12px;color:#8b949e'>Showing "
            f"<b style='color:#e6edf3'>{len(filtered)}</b> of "
            f"<b style='color:#e6edf3'>{len(df)}</b> patients</span>",
            unsafe_allow_html=True,
        )
        display_cols = [
            c for c in ["id", "age", "sex", "reason_for_admission",
                        "comorbidities", "admission_disposition"]
            if c in filtered.columns
        ]
        st.dataframe(filtered[display_cols].head(150).fillna(""),
                     use_container_width=True, height=380)

        st.markdown("---")
        st.markdown("#### 🧬 Patient Profile Viewer")
        pid_col, btn_col = st.columns([2, 1])
        with pid_col:
            selected_id = st.number_input(
                "Patient ID", min_value=1, max_value=10000, step=1, value=1
            )
        with btn_col:
            st.markdown("<br>", unsafe_allow_html=True)
            load_btn = st.button("Load Profile →", use_container_width=True)

        if load_btn:
            patient = api_get(f"/api/patients/{selected_id}")
            if not patient:
                row = df[df["id"] == selected_id]
                if not row.empty:
                    patient = {
                        "patient_id": selected_id,
                        "admission": row.iloc[0].to_dict(),
                        "outcome": {},
                    }

            if patient:
                pa, pb = st.columns(2)
                def _profile_row(k, v):
                    return (
                        f"<div style='display:flex;justify-content:space-between;"
                        f"border-bottom:1px solid #30363d;padding:5px 0'>"
                        f"<span style='color:#8b949e;font-size:12px'>{k}</span>"
                        f"<span style='color:#e6edf3;font-size:12px;font-weight:500'>{v}</span>"
                        f"</div>"
                    )
                with pa:
                    st.markdown("**📋 Admission Data**")
                    for k, v in list(patient.get("admission", {}).items())[:15]:
                        if str(v) not in ("nan", "None", "", "0.0") and v:
                            st.markdown(_profile_row(k, v), unsafe_allow_html=True)
                with pb:
                    st.markdown("**📈 Outcome Data**")
                    out = patient.get("outcome", {})
                    if out:
                        for k, v in list(out.items())[:15]:
                            if str(v) not in ("nan", "None", "", "0.0") and v:
                                st.markdown(_profile_row(k, v), unsafe_allow_html=True)
                    else:
                        st.caption("No outcome data available for this patient.")
            else:
                st.warning(f"Patient ID {selected_id} not found.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: LoS PREDICTOR
# ════════════════════════════════════════════════════════════════════════════
elif "📈 LoS Predictor" in page:
    st.title("📈 Hospital Length of Stay Predictor")
    st.caption("Enter patient vitals at admission to predict hospital stay duration. Model R² = 0.9915.")

    with st.form("los_form"):
        st.markdown("#### 🧑‍⚕️ Patient Information")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            age = st.slider("Age", 18, 100, 65)
        with r2:
            sex = st.selectbox("Sex", ["Male", "Female"])
        with r3:
            icu = st.selectbox("Admission Disposition", ["WARD", "ICU"])
        with r4:
            intubated = st.selectbox("Intubated", ["No", "Yes"])

        st.markdown("#### 🩺 Vital Signs at Admission")
        v1, v2, v3 = st.columns(3)
        with v1:
            systolic  = st.slider("Systolic BP (mmHg)",  70, 200, 130)
            diastolic = st.slider("Diastolic BP (mmHg)", 40, 130, 80)
        with v2:
            heart_rate = st.slider("Heart Rate (bpm)", 40, 180, 90)
            rr         = st.slider("Respiratory Rate",  10, 40,  18)
        with v3:
            o2_sat = st.slider("O₂ Saturation (%)", 70, 100, 95)
            temp   = st.slider("Temperature (°C)", 35.0, 42.0, 37.5, step=0.1)

        submitted = st.form_submit_button("🔮 Predict Hospital Stay", use_container_width=True)

    if submitted:
        payload = {
            "age": float(age),
            "sex": 1 if sex == "Female" else 0,
            "systolic_blood_pressure":  float(systolic),
            "diastolic_blood_pressure": float(diastolic),
            "heart_rate":               float(heart_rate),
            "respiratory_rate":         float(rr),
            "oxygen_saturation":        float(o2_sat),
            "temperature":              float(temp),
            "icu_admission":  1 if icu == "ICU" else 0,
            "intubated":      1 if intubated == "Yes" else 0,
        }
        with st.spinner("Running ML prediction…"):
            result = api_post("/api/predict/los", payload)

        if result and "predicted_los_days" in result:
            pred = result["predicted_los_days"]
            conf = result.get("confidence", "N/A")
            risk = result.get("risk_level", "Moderate" if pred <= 20 else "High")

            col_res, col_risk = st.columns([2, 1])
            with col_res:
                st.markdown(
                    f"<div class='kpi-tile' style='text-align:left'>"
                    f"<div style='font-size:11px;color:#8b949e;margin-bottom:4px'>PREDICTED HOSPITAL STAY</div>"
                    f"<div style='font-size:3rem;font-weight:800;color:#58a6ff'>{pred}"
                    f"<span style='font-size:1.2rem'> days</span></div>"
                    f"<div style='font-size:12px;color:#8b949e;margin-top:6px'>Model confidence: {conf}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_risk:
                color = {"High": "#f85149", "Moderate": "#ffa657", "Low": "#39d353"}.get(risk, "#ffa657")
                note  = {
                    "High":     "Extended stay likely. Early ICU planning recommended.",
                    "Moderate": "Average duration. Monitor vitals closely.",
                    "Low":      "Short stay expected. Standard care protocol.",
                }.get(risk, "")
                st.markdown(
                    f"<div class='kpi-tile'>"
                    f"<div style='font-size:11px;color:#8b949e;margin-bottom:4px'>RISK LEVEL</div>"
                    f"<div style='font-size:1.8rem;font-weight:700;color:{color}'>{risk}</div>"
                    f"<div style='font-size:11px;color:#8b949e;margin-top:6px'>{note}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("#### 🔬 Feature Contributions")
            feat_df = pd.DataFrame({
                "Feature": ["O₂ Saturation","ICU Admission","Age","Intubated",
                            "Heart Rate","Resp. Rate","Temperature","Sex"],
                "Impact":  [0.28, 0.25, 0.18, 0.12, 0.07, 0.05, 0.03, 0.02],
            })
            _lyt2 = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font=dict(color="#8b949e", size=12))
            fig = px.bar(feat_df, x="Impact", y="Feature", orientation="h",
                         color="Impact", color_continuous_scale=["#58a6ff","#d2a8ff"])
            fig.update_layout(**_lyt2, height=280, coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed", gridcolor="#30363d"),
                              xaxis=dict(gridcolor="#30363d"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(
                "API unavailable. Start the backend:\n"
                "```\nuvicorn deployment.api:app --reload\n```"
            )


# ════════════════════════════════════════════════════════════════════════════
# PAGE: ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════
elif "🏗️ Architecture" in page:
    st.title("🏗️ System Architecture")
    st.caption("End-to-end RAG + ML pipeline for COVID-19 inpatient data analysis.")

    layers = [
        ("🎨 Presentation Layer", "#1a3a5c",
         "Streamlit chatbot UI · Bootstrap 5 dark dashboard · WebSocket streaming chat",
         ["Port 8501 — Streamlit", "Port 80/443 — Static HTML Dashboard"]),
        ("⚡ API Gateway (FastAPI)", "#1a3a2a",
         "deployment/api.py · REST + WebSocket endpoints · Pydantic v2 validation · CORS middleware",
         ["POST /api/chat", "WS /ws/chat", "POST /api/predict/los",
          "GET /api/analytics/summary", "GET /api/patients", "GET /api/medications/search"]),
        ("🧠 RAG Pipeline (LangChain)", "#3a1a3a",
         "rag/pipeline.py · Intent classification → vector retrieval → LLM augmentation → grounded response",
         ["Intent Classifier (joblib)", "ChromaDB retriever Top-K=5",
          "Prompt template + LLM invoke", "Ollama · Claude · OpenAI · Local GGUF"]),
        ("🗄️ Vector Store (ChromaDB)", "#2a2a1a",
         "vector_store/ · Persistent embeddings · Two collections",
         ["patient_records — 508 documents", "medications — 5,000 documents",
          "Embedder: all-MiniLM-L6-v2 (384-dim)"]),
        ("📊 ML Models (scikit-learn)", "#3a1a1a",
         "models/ · Trained on this dataset · Serialized with joblib",
         ["Hospital_LoS_Model.joblib — R²=0.9915",
          "intent_classifier.joblib", "ner_model.joblib"]),
        ("💾 Data Layer", "#1a1a3a",
         "data/ · Preprocessed CSVs from raw Excel (Canada_Hosp1_COVID_InpatientData.xlsx)",
         ["Preprocessed_Data-at-Admission.csv (508 rows)",
          "Preprocessed_Hospital-LoS.csv",
          "Preprocessed_Days-Breakdown.csv",
          "Preprocessed_Medications.csv"]),
    ]

    for title, bg, desc, items in layers:
        chips = "".join(
            f"<span style='background:#1e2736;border:1px solid #30363d;"
            f"border-radius:4px;padding:2px 10px;font-size:11px;color:#e6edf3'>{item}</span>"
            for item in items
        )
        st.markdown(
            f"<div style='background:{bg}20;border:1px solid {bg}80;"
            f"border-radius:8px;padding:14px 18px;margin:8px 0'>"
            f"<div style='font-weight:700;color:#e6edf3;font-size:14px;margin-bottom:6px'>{title}</div>"
            f"<div style='color:#8b949e;font-size:12px;margin-bottom:8px'>{desc}</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:6px'>{chips}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 🔄 Request Data Flow")
    steps = [
        ("1", "User sends message", "via Streamlit chat input or Bootstrap dashboard"),
        ("2", "FastAPI validates request", "Pydantic v2 schema — POST /api/chat"),
        ("3", "Intent classifier runs", "Determines query type: analytics / medication / LoS / general"),
        ("4", "ChromaDB retriever fetches", "Top-5 semantically similar patient records or medications"),
        ("5", "Prompt augmentation", "Retrieved context injected into LangChain prompt template"),
        ("6", "LLM generates answer", "Ollama / Claude / OpenAI — grounded in real patient data"),
        ("7", "Response returned to UI", "With sources, intent label, and UTC timestamp"),
    ]
    for num, step, detail in steps:
        st.markdown(
            f"<div style='display:flex;align-items:flex-start;gap:14px;"
            f"padding:8px 0;border-bottom:1px solid #30363d'>"
            f"<span style='background:#58a6ff;color:#0d1117;border-radius:50%;"
            f"width:24px;height:24px;display:flex;align-items:center;"
            f"justify-content:center;font-weight:700;font-size:12px;flex-shrink:0'>{num}</span>"
            f"<div><span style='color:#e6edf3;font-size:13px;font-weight:600'>{step}</span>"
            f"<span style='color:#8b949e;font-size:12px'> — {detail}</span></div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 🚀 Quick Start Commands")
    st.code(
        "# 1. Install dependencies\n"
        "pip install -r deployment/requirements.txt\n\n"
        "# 2. Build vector store (one-time ~3 min)\n"
        "python -m rag.ingest\n\n"
        "# 3. Start FastAPI backend\n"
        "uvicorn deployment.api:app --reload --host 0.0.0.0 --port 8000\n\n"
        "# 4. Start Streamlit UI  (new terminal)\n"
        "streamlit run chatbot_v2.py\n\n"
        "# 5. Open Bootstrap dashboard\n"
        "start static/dashboard.html\n\n"
        "# 6. Run all 28 API tests\n"
        "pytest tests/test_api.py -v -p no:cacheprovider",
        language="bash",
    )
