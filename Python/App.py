import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="AuditMind Intelligence Ltd", layout="wide")

# -----------------------------
# SESSION STATE
# -----------------------------
defaults = {
    "logged_in": False,
    "page": "Home",
    "results_table": None,
    "flagged_only": None,
    "rules_fired": None,
    "current_role": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -----------------------------
# STREAMLIT RERUN COMPATIBILITY
# -----------------------------
def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# -----------------------------
# NAVIGATION HELPERS
# -----------------------------
def go_home():
    st.session_state.page = "Home"
    safe_rerun()

def go_sme_upload():
    st.session_state.page = "SME Upload"
    st.session_state.current_role = "SME"
    safe_rerun()

def go_acc_upload():
    st.session_state.page = "Accountant Upload"
    st.session_state.current_role = "Accountant"
    safe_rerun()

def do_logout():
    st.session_state.logged_in = False
    st.session_state.page = "Home"
    st.session_state.results_table = None
    st.session_state.flagged_only = None
    st.session_state.rules_fired = None
    st.session_state.current_role = None
    safe_rerun()

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    color: #111827;
    font-family: Arial, sans-serif;
}
html, body, [class*="css"] {
    font-family: Arial, sans-serif;
    color: #111827 !important;
}
section[data-testid="stSidebar"] {
    display: none !important;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
h1, h2, h3, h4, h5, h6, p, label, li, span, div {
    color: #111827 !important;
}
.logo-circle {
    width: 72px;
    height: 72px;
    margin: 0 auto 14px auto;
    border-radius: 18px;
    background: linear-gradient(135deg, #0f766e, #1d4ed8);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white !important;
    font-size: 30px;
    font-weight: 700;
    box-shadow: 0 12px 30px rgba(29, 78, 216, 0.18);
}
.hero-box {
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    padding: 28px;
    border-radius: 20px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    margin-bottom: 20px;
}
.home-main-title {
    font-size: 26px !important;
    font-weight: 700;
    margin-bottom: 10px;
    color: #111827 !important;
    text-align: center;
}
.small-note {
    color: #6b7280 !important;
    font-size: 14px;
    line-height: 1.6;
    text-align: center;
}
.login-helper {
    text-align: center;
    font-size: 13px;
    color: #6b7280 !important;
    margin-top: 10px;
}
.info-card {
    background: #ffffff;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.04);
    margin-bottom: 16px;
}
.summary-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.04);
}
.rules-card {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 18px;
    padding: 20px;
    margin-top: 24px;
    margin-bottom: 18px;
    box-shadow: 0 4px 14px rgba(14, 165, 233, 0.06);
}
.confidence-card {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 18px;
    padding: 20px;
    margin-top: 24px;
    margin-bottom: 18px;
    box-shadow: 0 4px 14px rgba(34, 197, 94, 0.06);
}
.stButton > button, .stDownloadButton > button, [data-testid="baseButton-secondary"] {
    border-radius: 12px !important;
    border: 1px solid #d1d5db !important;
    padding-top: 0.55rem !important;
    padding-bottom: 0.55rem !important;
    font-weight: 600 !important;
}
div[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    padding: 12px;
    border-radius: 14px;
    color: #111827 !important;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.03);
}
.badge-high {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background-color: #fee2e2;
    color: #b91c1c !important;
    font-weight: 700;
    font-size: 13px;
}
.badge-medium {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background-color: #ffedd5;
    color: #c2410c !important;
    font-weight: 700;
    font-size: 13px;
}
.badge-low {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background-color: #dcfce7;
    color: #15803d !important;
    font-weight: 700;
    font-size: 13px;
}
.badge-confidence-high {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    background-color: #fee2e2;
    color: #b91c1c !important;
    font-weight: 600;
    font-size: 12px;
}
.badge-confidence-medium {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    background-color: #ffedd5;
    color: #c2410c !important;
    font-weight: 600;
    font-size: 12px;
}
.badge-confidence-low {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    background-color: #dcfce7;
    color: #15803d !important;
    font-weight: 600;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HELPERS
# -----------------------------
def render_logo():
    st.markdown('<div class="logo-circle">A</div>', unsafe_allow_html=True)

def risk_badge_text(level):
    if level == "High":
        return "🔴 High"
    elif level == "Medium":
        return "🟠 Medium"
    return "🟢 Low"

def risk_badge_html(level):
    if level == "High":
        return '<span class="badge-high">🔴 High</span>'
    elif level == "Medium":
        return '<span class="badge-medium">🟠 Medium</span>'
    return '<span class="badge-low">🟢 Low</span>'

def add_risk_badges(df):
    df = df.copy()
    df["Risk_Badge"] = df["Risk_Level"].apply(risk_badge_text)
    return df

def confidence_label(score):
    """Convert raw confidence score (0-1) to human-readable label."""
    if score >= 0.70:
        return "🔴 High"
    elif score >= 0.50:
        return "🟠 Medium"
    return "🟢 Low"

# -----------------------------
# ★ NEW: RULEBOOK (from Script 2)
# Each rule has an ID, description, condition function, and risk category.
# To add a new rule simply append a new dict to this list.
# -----------------------------
RULEBOOK = [
    {
        "id": "CIS-001",
        "description": "Subcontractor invoice over £5,000",
        "condition": lambda r: (
            str(r.get("Supplier_Type", "")).strip().lower() == "subcontractor"
            and float(r.get("Amount", 0)) > 5000
        ),
        "risk": "CIS Risk — high-value subcontractor",
        "severity": "High",
    },
    {
        "id": "CIS-002",
        "description": "Any subcontractor payment (CIS registration check required)",
        "condition": lambda r: (
            str(r.get("Supplier_Type", "")).strip().lower() == "subcontractor"
        ),
        "risk": "CIS Risk — verify registration status",
        "severity": "Medium",
    },
    {
        "id": "DUP-001",
        "description": "Duplicate Invoice ID and Amount detected",
        "condition": lambda r: r.get("duplicate", 0) == 1,
        "risk": "Duplicate transaction — possible double payment",
        "severity": "High",
    },
    {
        "id": "VAT-001",
        "description": "Invalid or missing VAT code",
        "condition": lambda r: r.get("vat_issue", 0) == 1,
        "risk": "VAT inconsistency — code not in Standard / Reverse / Zero",
        "severity": "Medium",
    },
    {
        "id": "ANO-001",
        "description": "ML anomaly: unusual transaction amount",
        "condition": lambda r: r.get("high_anomaly", 0) == 1,
        "risk": "Statistical anomaly — amount deviates significantly from peers",
        "severity": "High",
    },
    {
        "id": "ANO-002",
        "description": "High ML confidence anomaly score (≥ 0.70)",
        "condition": lambda r: float(r.get("confidence_score", 0)) >= 0.70,
        "risk": "High confidence anomaly — model strongly flags this transaction",
        "severity": "High",
    },
]


def apply_rulebook(df):
    """
    Apply every rule in RULEBOOK to every row in df.
    Returns a DataFrame listing every rule firing event.
    """
    results = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        for rule in RULEBOOK:
            try:
                if rule["condition"](row_dict):
                    results.append({
                        "Invoice_ID":   row_dict.get("Invoice_ID", ""),
                        "Amount":       row_dict.get("Amount", 0),
                        "Rule_ID":      rule["id"],
                        "Description":  rule["description"],
                        "Risk_Detail":  rule["risk"],
                        "Severity":     rule["severity"],
                    })
            except Exception:
                pass
    if results:
        return pd.DataFrame(results)
    return pd.DataFrame(
        columns=["Invoice_ID", "Amount", "Rule_ID", "Description", "Risk_Detail", "Severity"]
    )


# -----------------------------
# DATA PROCESSING
# (Script 1 engine + Script 2 confidence score added)
# -----------------------------
def process_uploaded_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    required_cols = ["Amount", "Invoice_ID", "VAT_Code", "Supplier_Type"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return None, None, None

    df["Amount"]        = pd.to_numeric(df["Amount"], errors="coerce")
    df                  = df.dropna(subset=["Amount"]).copy()
    df["Invoice_ID"]    = df["Invoice_ID"].astype(str).str.strip()
    df["VAT_Code"]      = df["VAT_Code"].astype(str).str.strip()
    df["Supplier_Type"] = df["Supplier_Type"].astype(str).str.strip()

    # ── Feature engineering ──────────────────────────────────────────
    df["log_amount"] = np.log1p(df["Amount"])

    # ── Anomaly detection (Script 1) + Confidence score (Script 2) ──
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(df[["Amount", "log_amount"]])

    df["anomaly_raw"]      = model.predict(df[["Amount", "log_amount"]])
    df["high_anomaly"]     = df["anomaly_raw"].apply(lambda x: 1 if x == -1 else 0)

    # ★ NEW: confidence_score — higher means the model is MORE certain
    # it is an anomaly. Clipped to [0, 1] range for display clarity.
    raw_scores             = model.decision_function(df[["Amount", "log_amount"]])
    df["confidence_score"] = np.clip(1 - raw_scores, 0, 1)
    df["Confidence_Score"] = df["confidence_score"].round(4)
    df["Confidence_Level"] = df["confidence_score"].apply(confidence_label)

    # ── Rule-based checks (Script 1) ────────────────────────────────
    df["duplicate"] = df.duplicated(
        subset=["Invoice_ID", "Amount"], keep=False
    ).astype(int)

    valid_vat_codes = ["Standard", "Reverse", "Zero"]
    df["vat_issue"] = (~df["VAT_Code"].isin(valid_vat_codes)).astype(int)

    df["cis_issue"] = (
        df["Supplier_Type"].str.strip().str.lower() == "subcontractor"
    ).astype(int)

    # ── Composite risk scoring (Script 1 weights) ────────────────────
    df["risk_score"] = (
        df["high_anomaly"] * 2 +
        df["duplicate"]    * 2 +
        df["vat_issue"]    * 1 +
        df["cis_issue"]    * 1
    )

    def classify_risk(score):
        if score >= 4:
            return "High"
        elif score >= 2:
            return "Medium"
        return "Low"

    df["Risk_Level"] = df["risk_score"].apply(classify_risk)

    # ── Human-readable flag columns ──────────────────────────────────
    df["High_Anomaly"] = df["high_anomaly"].map({1: "Yes", 0: "No"})
    df["Duplicate"]    = df["duplicate"].map({1: "Yes", 0: "No"})
    df["VAT_Issue"]    = df["vat_issue"].map({1: "Yes", 0: "No"})
    df["CIS_Issue"]    = df["cis_issue"].map({1: "Yes", 0: "No"})

    # ── Explanation (Script 1) ───────────────────────────────────────
    def explain(row):
        reasons = []
        if row["high_anomaly"] == 1: reasons.append("High anomaly detected")
        if row["duplicate"]    == 1: reasons.append("Duplicate transaction")
        if row["vat_issue"]    == 1: reasons.append("VAT inconsistency")
        if row["cis_issue"]    == 1: reasons.append("CIS risk")
        return ", ".join(reasons) if reasons else "No risk"

    df["Explanation"] = df.apply(explain, axis=1)

    # ── Recommended actions (Script 1) ──────────────────────────────
    def recommend_action(row):
        actions = []
        if row["duplicate"]    == 1: actions.append("Check for duplicate payment before approval")
        if row["vat_issue"]    == 1: actions.append("Verify VAT code against transaction type")
        if row["cis_issue"]    == 1: actions.append("Review subcontractor CIS compliance and deductions")
        if row["high_anomaly"] == 1: actions.append("Investigate unusual transaction amount or pattern")
        return " | ".join(actions) if actions else "No action needed"

    df["Recommended_Action"] = df.apply(recommend_action, axis=1)

    # ── Build results table ──────────────────────────────────────────
    results_table = df[[
        "Invoice_ID",
        "Amount",
        "Risk_Level",
        "High_Anomaly",
        "Duplicate",
        "VAT_Issue",
        "CIS_Issue",
        "Confidence_Score",   # ★ NEW
        "Confidence_Level",   # ★ NEW
        "Explanation",
        "Recommended_Action",
    ]].copy()

    results_table["Flag_Count"] = (
        (results_table["High_Anomaly"] == "Yes").astype(int) +
        (results_table["Duplicate"]    == "Yes").astype(int) +
        (results_table["VAT_Issue"]    == "Yes").astype(int) +
        (results_table["CIS_Issue"]    == "Yes").astype(int)
    )

    results_table = results_table.sort_values(
        by=["Flag_Count", "Confidence_Score", "Amount"],
        ascending=[False, False, False]
    ).drop(columns=["Flag_Count"])

    flagged_only = results_table[
        (results_table["High_Anomaly"] == "Yes") |
        (results_table["Duplicate"]    == "Yes") |
        (results_table["VAT_Issue"]    == "Yes") |
        (results_table["CIS_Issue"]    == "Yes")
    ].copy()

    # ── Apply rulebook ───────────────────────────────────────────────
    rules_fired = apply_rulebook(df)

    return results_table, flagged_only, rules_fired


# ═══════════════════════════════════════════════════════════
# LOGIN PAGE
# ═══════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    render_logo()
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.markdown("""
        <div class="hero-box">
            <div class="home-main-title">AuditMind Intelligence Ltd</div>
            <p class="small-note">
                AuditMind helps accountants and construction businesses identify financial risk faster
                using anomaly detection, VAT inconsistency checks, CIS checks, duplicate invoice detection,
                and a structured compliance rulebook.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username  = st.text_input("Username")
            password  = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                if username == "admin" and password == "secure123":
                    st.session_state.logged_in = True
                    st.session_state.page      = "Home"
                    safe_rerun()
                else:
                    st.error("Invalid username or password")

        st.markdown(
            '<div class="login-helper">Demo login: admin / secure123</div>',
            unsafe_allow_html=True,
        )
    st.stop()


# ═══════════════════════════════════════════════════════════
# TOP BAR
# ═══════════════════════════════════════════════════════════
top_left, top_mid, top_right = st.columns([8, 1, 1])
with top_right:
    st.button("Logout", key="logout_btn", use_container_width=True, on_click=do_logout)

st.title("AuditMind Intelligence Ltd")


# ═══════════════════════════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════════════════════════
if st.session_state.page == "Home":
    render_logo()

    st.markdown("""
    <div class="hero-box">
        <div class="home-main-title">AI Financial Risk Intelligence Platform</div>
        <p class="small-note">
            AuditMind Intelligence Ltd helps accountants and construction businesses identify financial risk faster
            using anomaly detection, VAT inconsistency checks, CIS checks, duplicate invoice detection,
            confidence scoring, and a structured compliance rulebook.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>What the platform does</h3>
            <ul>
                <li>Upload transaction data in CSV format</li>
                <li>Detect high anomalies using Isolation Forest ML</li>
                <li>Show ML confidence score per transaction</li>
                <li>Identify VAT inconsistencies</li>
                <li>Identify CIS-related subcontractor risks</li>
                <li>Detect duplicate invoice patterns</li>
                <li>Apply structured compliance rulebook (CIS-001, DUP-001 etc.)</li>
                <li>Classify risk levels: High / Medium / Low</li>
                <li>Recommend corrective actions</li>
                <li>Download final results for reporting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>Who it is for</h3>
            <ul>
                <li>SME Business Owners</li>
                <li>Accountants</li>
                <li>Construction companies</li>
                <li>Internal finance teams</li>
                <li>Compliance reviewers</li>
            </ul>
            <p class="small-note">Choose a dashboard below to continue.</p>
        </div>
        """, unsafe_allow_html=True)

    btn_left, btn_right = st.columns(2)
    with btn_left:
        st.button("SME Owner Dashboard",    use_container_width=True, key="sme_home_btn",  on_click=go_sme_upload)
    with btn_right:
        st.button("Accountant Dashboard",   use_container_width=True, key="acc_home_btn",  on_click=go_acc_upload)


# ═══════════════════════════════════════════════════════════
# SME UPLOAD PAGE
# ═══════════════════════════════════════════════════════════
elif st.session_state.page == "SME Upload":
    col1, col2 = st.columns([1, 6])
    with col1:
        st.button("← Back", key="back_home_sme", use_container_width=True, on_click=go_home)

    st.subheader("SME Owner Upload Page")
    st.write("Upload your transaction CSV. Results will appear on the next page.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="sme_upload")

    st.markdown("""
    **Required columns**
    - `Amount`
    - `Invoice_ID`
    - `VAT_Code`
    - `Supplier_Type`
    """)

    if uploaded_file is not None:
        results_table, flagged_only, rules_fired = process_uploaded_data(uploaded_file)
        if results_table is not None:
            st.session_state.results_table = results_table
            st.session_state.flagged_only  = flagged_only
            st.session_state.rules_fired   = rules_fired
            st.success("File processed. Redirecting to results…")
            st.session_state.page = "SME Results"
            safe_rerun()


# ═══════════════════════════════════════════════════════════
# SME RESULTS PAGE
# ═══════════════════════════════════════════════════════════
elif st.session_state.page == "SME Results":
    col1, col2 = st.columns([1, 6])
    with col1:
        st.button("← Back", key="back_sme_upload", use_container_width=True, on_click=go_sme_upload)

    st.subheader("SME Owner Results")

    if st.session_state.results_table is None:
        st.warning("No data found. Please upload a file first.")
    else:
        results_table = st.session_state.results_table
        flagged_only  = st.session_state.flagged_only
        rules_fired   = st.session_state.rules_fired

        high_count   = (results_table["Risk_Level"] == "High").sum()
        medium_count = (results_table["Risk_Level"] == "Medium").sum()
        low_count    = (results_table["Risk_Level"] == "Low").sum()
        flagged_count = len(flagged_only)

        # ── Metrics ──────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("High Risk",     high_count)
        c2.metric("Medium Risk",   medium_count)
        c3.metric("Low Risk",      low_count)
        c4.metric("Total Flagged", flagged_count)

        # ── Risk badges ───────────────────────────────────────────────
        st.markdown("<div class='summary-card'>", unsafe_allow_html=True)
        st.subheader("Risk Overview")
        b1, b2, b3 = st.columns(3)
        with b1: st.markdown(risk_badge_html("High"),   unsafe_allow_html=True)
        with b2: st.markdown(risk_badge_html("Medium"), unsafe_allow_html=True)
        with b3: st.markdown(risk_badge_html("Low"),    unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Risk chart ────────────────────────────────────────────────
        risk_chart_df = pd.DataFrame({
            "Risk_Level": ["High", "Medium", "Low"],
            "Count":      [high_count, medium_count, low_count],
        })
        fig = px.bar(risk_chart_df, x="Risk_Level", y="Count",
                     title="Risk Level Summary",
                     color="Risk_Level",
                     color_discrete_map={"High": "#ef4444", "Medium": "#f97316", "Low": "#22c55e"})
        st.plotly_chart(fig, use_container_width=True)

        # ── Flagged transactions (simplified SME view) ────────────────
        st.subheader("Flagged Transactions for SME Review")
        sme_view = flagged_only[[
            "Invoice_ID", "Amount", "Risk_Level",
            "Confidence_Score", "Confidence_Level",   # ★ NEW
            "Explanation", "Recommended_Action",
        ]].copy()
        sme_view = add_risk_badges(sme_view)
        sme_view = sme_view[[
            "Invoice_ID", "Amount", "Risk_Badge",
            "Confidence_Score", "Confidence_Level",
            "Explanation", "Recommended_Action",
        ]]
        st.dataframe(sme_view, use_container_width=True)

        # ── ★ NEW: Rules Fired table (SME version — simplified) ───────
        if not rules_fired.empty:
            st.markdown("<div class='rules-card'>", unsafe_allow_html=True)
            st.subheader("📋 Compliance Rules Fired")
            st.caption(
                "These are the specific compliance rules triggered by your data. "
                "Each rule ID maps to a defined regulatory or fraud-prevention check."
            )
            sme_rules = rules_fired[[
                "Invoice_ID", "Amount", "Rule_ID", "Risk_Detail", "Severity"
            ]].copy()
            st.dataframe(sme_rules, use_container_width=True)

            rules_summary = rules_fired.groupby("Rule_ID").size().reset_index(name="Times_Fired")
            fig_rules = px.bar(
                rules_summary, x="Rule_ID", y="Times_Fired",
                title="Rules Fired — Frequency",
                color="Rule_ID",
            )
            st.plotly_chart(fig_rules, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Download ──────────────────────────────────────────────────
        csv_sme = flagged_only[[
            "Invoice_ID", "Amount", "Risk_Level",
            "Confidence_Score", "Confidence_Level",
            "Explanation", "Recommended_Action",
        ]].to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download SME Risk Summary CSV",
            data=csv_sme,
            file_name="auditmind_sme_summary.csv",
            mime="text/csv",
        )


# ═══════════════════════════════════════════════════════════
# ACCOUNTANT UPLOAD PAGE
# ═══════════════════════════════════════════════════════════
elif st.session_state.page == "Accountant Upload":
    col1, col2 = st.columns([1, 6])
    with col1:
        st.button("← Back", key="back_home_acc", use_container_width=True, on_click=go_home)

    st.subheader("Accountant Upload Page")
    st.write("Upload client transaction data. Full risk detail appears on the next page.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="acc_upload")

    st.markdown("""
    **Required columns**
    - `Amount`
    - `Invoice_ID`
    - `VAT_Code`
    - `Supplier_Type`
    """)

    if uploaded_file is not None:
        results_table, flagged_only, rules_fired = process_uploaded_data(uploaded_file)
        if results_table is not None:
            st.session_state.results_table = results_table
            st.session_state.flagged_only  = flagged_only
            st.session_state.rules_fired   = rules_fired
            st.success("File processed. Redirecting to results…")
            st.session_state.page = "Accountant Results"
            safe_rerun()


# ═══════════════════════════════════════════════════════════
# ACCOUNTANT RESULTS PAGE
# ═══════════════════════════════════════════════════════════
elif st.session_state.page == "Accountant Results":
    col1, col2 = st.columns([1, 6])
    with col1:
        st.button("← Back", key="back_acc_upload", use_container_width=True, on_click=go_acc_upload)

    st.subheader("Accountant Results — Full Detail")

    if st.session_state.results_table is None:
        st.warning("No data found. Please upload a file first.")
    else:
        results_table = st.session_state.results_table
        flagged_only  = st.session_state.flagged_only
        rules_fired   = st.session_state.rules_fired

        # ── Metrics ──────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("High Anomalies", (results_table["High_Anomaly"] == "Yes").sum())
        c2.metric("Duplicates",     (results_table["Duplicate"]    == "Yes").sum())
        c3.metric("VAT Issues",     (results_table["VAT_Issue"]    == "Yes").sum())
        c4.metric("CIS Issues",     (results_table["CIS_Issue"]    == "Yes").sum())

        # ── Confidence score summary metrics ─────────────────────────
        st.markdown("<div class='confidence-card'>", unsafe_allow_html=True)
        st.subheader("🎯 ML Confidence Score Summary")
        st.caption(
            "Confidence Score reflects how strongly the Isolation Forest model flags each "
            "transaction as anomalous. Scores closer to 1.0 indicate higher certainty of anomaly."
        )
        conf_h = (results_table["Confidence_Level"] == "🔴 High").sum()
        conf_m = (results_table["Confidence_Level"] == "🟠 Medium").sum()
        conf_l = (results_table["Confidence_Level"] == "🟢 Low").sum()
        avg_conf = results_table["Confidence_Score"].mean()

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("High Confidence Anomalies",   conf_h)
        cc2.metric("Medium Confidence Anomalies", conf_m)
        cc3.metric("Low Confidence (Normal)",     conf_l)
        cc4.metric("Avg Confidence Score",        f"{avg_conf:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Risk legend ───────────────────────────────────────────────
        st.markdown("<div class='summary-card'>", unsafe_allow_html=True)
        st.subheader("Risk Legend")
        r1, r2, r3 = st.columns(3)
        with r1: st.markdown(risk_badge_html("High"),   unsafe_allow_html=True)
        with r2: st.markdown(risk_badge_html("Medium"), unsafe_allow_html=True)
        with r3: st.markdown(risk_badge_html("Low"),    unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Full detailed results table ───────────────────────────────
        st.subheader("Detailed Results Table")
        detailed_view = add_risk_badges(results_table)
        detailed_view = detailed_view[[
            "Invoice_ID", "Amount", "Risk_Badge",
            "High_Anomaly", "Duplicate", "VAT_Issue", "CIS_Issue",
            "Confidence_Score", "Confidence_Level",    # ★ NEW
            "Explanation", "Recommended_Action",
        ]]
        st.dataframe(detailed_view, use_container_width=True)

        # ── Flagged only table ────────────────────────────────────────
        st.subheader("Flagged Transactions Only")
        flagged_view = add_risk_badges(flagged_only)
        flagged_view = flagged_view[[
            "Invoice_ID", "Amount", "Risk_Badge",
            "High_Anomaly", "Duplicate", "VAT_Issue", "CIS_Issue",
            "Confidence_Score", "Confidence_Level",
            "Explanation", "Recommended_Action",
        ]]
        st.dataframe(flagged_view, use_container_width=True)

        # ── Charts ────────────────────────────────────────────────────
        chart_df = pd.DataFrame({
            "Risk_Type": ["High Anomaly", "Duplicate", "VAT Issue", "CIS Issue"],
            "Count": [
                (results_table["High_Anomaly"] == "Yes").sum(),
                (results_table["Duplicate"]    == "Yes").sum(),
                (results_table["VAT_Issue"]    == "Yes").sum(),
                (results_table["CIS_Issue"]    == "Yes").sum(),
            ],
        })
        fig_risk = px.bar(
            chart_df, x="Risk_Type", y="Count",
            title="Detected Risk Type Summary",
            color="Risk_Type",
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        # ── ★ NEW: Confidence Score Distribution Chart ─────────────────
        fig_conf = px.histogram(
            results_table,
            x="Confidence_Score",
            nbins=20,
            title="ML Confidence Score Distribution",
            labels={"Confidence_Score": "Confidence Score (0 = normal, 1 = strong anomaly)"},
            color_discrete_sequence=["#6366f1"],
        )
        fig_conf.add_vline(x=0.70, line_dash="dash", line_color="red",
                           annotation_text="High threshold (0.70)",
                           annotation_position="top right")
        fig_conf.add_vline(x=0.50, line_dash="dash", line_color="orange",
                           annotation_text="Medium threshold (0.50)",
                           annotation_position="top right")
        st.plotly_chart(fig_conf, use_container_width=True)

        # ── ★ NEW: Rulebook — Rules Fired Table ───────────────────────
        st.markdown("<div class='rules-card'>", unsafe_allow_html=True)
        st.subheader("📋 Compliance Rulebook — Rules Fired")
        st.caption(
            "Every rule in the AuditMind rulebook is evaluated against each transaction. "
            "The table below shows which rules fired, how many times, and for which invoices."
        )

        if rules_fired.empty:
            st.info("No compliance rules were triggered by this dataset.")
        else:
            st.dataframe(rules_fired, use_container_width=True)

            # Rules frequency summary
            st.subheader("Rules Summary")
            rules_summary = (
                rules_fired.groupby(["Rule_ID", "Description", "Severity"])
                .size()
                .reset_index(name="Times_Fired")
                .sort_values("Times_Fired", ascending=False)
            )
            st.dataframe(rules_summary, use_container_width=True)

            fig_rules = px.bar(
                rules_summary,
                x="Rule_ID",
                y="Times_Fired",
                color="Severity",
                title="Compliance Rules Fired — Frequency by Rule",
                color_discrete_map={"High": "#ef4444", "Medium": "#f97316", "Low": "#22c55e"},
                text="Times_Fired",
            )
            fig_rules.update_traces(textposition="outside")
            st.plotly_chart(fig_rules, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Downloads ─────────────────────────────────────────────────
        dl1, dl2 = st.columns(2)

        with dl1:
            csv_full = results_table.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Full Accountant Results CSV",
                data=csv_full,
                file_name="auditmind_accountant_full.csv",
                mime="text/csv",
            )

        with dl2:
            if not rules_fired.empty:
                csv_rules = rules_fired.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Rules Fired CSV",
                    data=csv_rules,
                    file_name="auditmind_rules_fired.csv",
                    mime="text/csv",
                )