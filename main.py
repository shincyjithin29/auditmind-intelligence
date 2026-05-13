# ============================================================
# AuditMind Intelligence Ltd — main.py v3.2.1
# ============================================================
# LOCAL:   python -m uvicorn main:app --reload --port 8000
# RENDER:  python -m uvicorn main:app --host 0.0.0.0 --port 8000
#
# PAGES:
#   GET  /            -> Home.html
#   GET  /login       -> login.html
#   GET  /upload      -> upload.html
#   GET  /results     -> results.html
#
# API:
#   GET  /api/health  -> health check (GET + HEAD)
#   POST /api/login   -> authenticate user
#   POST /api/register -> register new company user
#   POST /api/columns -> read CSV headers for column mapping
#   POST /api/analyse -> run full risk analysis (+ FSI)
#
# CHANGES IN v3.2.1:
#   - Replaced passlib with direct bcrypt — fixes registration
#   - Health endpoint now accepts HEAD requests (UptimeRobot)
#   - All v3.2.0 features retained unchanged
# ============================================================

import os
import io
import re
import traceback
import logging
import bcrypt

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# ── Structured logging ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("auditmind")

# ── Supabase client ───────────────────────────────────────────
try:
    from supabase import create_client, Client as SupabaseClient
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger.warning("supabase package not installed — registration disabled")

# ── Password hashing — direct bcrypt (no passlib) ────────────
def hash_password(password: str) -> str:
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            hashed.encode("utf-8")
        )
    except Exception:
        return False

# ── Supabase setup ────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

def get_supabase():
    if not SUPABASE_AVAILABLE:
        raise RuntimeError("Supabase not available")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Supabase credentials not configured")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Folder where main.py lives ────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(
    title="AuditMind Intelligence — Decision Support API",
    version="3.2.1",
    docs_url=None,
    redoc_url=None,
)

ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "https://auditmind-intelligence.onrender.com",
    "https://www.auditmindintelligence.co.uk",
    "https://auditmindintelligence.co.uk",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "HEAD"],
    allow_headers=["Content-Type", "Authorization"],
)

# ============================================================
# ADMIN CREDENTIALS — fallback always works
# ============================================================
ADMIN_PASSWORD = os.getenv("AUDITMIND_ADMIN_PASSWORD", "secure123")

# ============================================================
# FLEXIBLE SUBCONTRACTOR DETECTION
# ============================================================
def is_subcontractor(value: str) -> bool:
    cleaned = re.sub(r'[\s\-_]', '', str(value).strip().lower())
    return cleaned == 'subcontractor'

# ============================================================
# COMPLIANCE RULEBOOK — v2025.06.v1
# ============================================================
RULEBOOK_VERSION = "2025.06.v1"

RULEBOOK = [
    {
        "id":          "CIS-001",
        "description": "Subcontractor invoice over £5,000",
        "condition":   lambda r: (
            is_subcontractor(str(r.get("Supplier_Type", "")))
            and float(r.get("Amount", 0)) > 5000
        ),
        "risk":     "CIS Risk — high-value subcontractor payment requires HMRC verification",
        "severity": "High",
    },
    {
        "id":          "CIS-002",
        "description": "Any subcontractor payment — CIS registration check required",
        "condition":   lambda r: is_subcontractor(str(r.get("Supplier_Type", ""))),
        "risk":     "CIS Risk — verify subcontractor registration status with HMRC",
        "severity": "Medium",
    },
    {
        "id":          "DUP-001",
        "description": "Possible duplicate — same amount and supplier type",
        "condition":   lambda r: r.get("duplicate", 0) == 1,
        "risk":     "Duplicate transaction — same amount from same supplier type, possible double payment",
        "severity": "High",
    },
    {
        "id":          "VAT-001",
        "description": "Invalid or unrecognised VAT code",
        "condition":   lambda r: r.get("vat_issue", 0) == 1,
        "risk":     "VAT inconsistency — code is not Standard, Reverse, or Zero",
        "severity": "Medium",
    },
    {
        "id":          "ANO-001",
        "description": "Statistical outlier — unusual transaction amount",
        "condition":   lambda r: r.get("high_anomaly", 0) == 1,
        "risk":     "Statistical anomaly — amount deviates significantly from peer transactions",
        "severity": "High",
    },
    {
        "id":          "ANO-002",
        "description": "Strong statistical outlier — anomaly score 0.70 or above",
        "condition":   lambda r: float(r.get("confidence_score", 0)) >= 0.70,
        "risk":     "High anomaly score — model strongly flags this transaction for review",
        "severity": "High",
    },
    {
        "id":          "RND-001",
        "description": "Large round-number amount (£1,000 or above, divisible by 1,000)",
        "condition":   lambda r: (
            float(r.get("Amount", 0)) >= 1000
            and float(r.get("Amount", 0)) % 1000 == 0
        ),
        "risk":     "Round number alert — fabricated invoices often use round amounts",
        "severity": "Medium",
    },
    {
        "id":          "HVL-001",
        "description": "Extreme value — above 10x dataset average",
        "condition":   lambda r: float(r.get("amount_ratio", 0)) > 10,
        "risk":     "Extreme value — transaction is more than 10x the dataset average",
        "severity": "High",
    },
]

# ============================================================
# VAT NORMALISATION MAP
# ============================================================
VAT_MAP = {
    "t1": "Standard", "output2": "Standard", "output": "Standard",
    "20%vat": "Standard", "20% vat": "Standard", "sr": "Standard",
    "standardrated": "Standard", "vat20": "Standard",
    "t2": "Reverse", "rc": "Reverse", "reversecharge": "Reverse",
    "domesticreversecharge": "Reverse",
    "t0": "Zero", "z": "Zero", "zerorated": "Zero",
    "zeroratedexpenses": "Zero", "0%": "Zero",
    "exempt": "Exempt", "e": "Exempt",
    "input": "Input",
}

def normalise_vat(code: str) -> str:
    cleaned = re.sub(r'[\s\-_]', '', str(code).strip().lower())
    return VAT_MAP.get(cleaned, str(code).strip())

# ============================================================
# HELPERS
# ============================================================

def read_html(filename: str) -> str:
    filepath = os.path.join(BASE, filename)
    if not os.path.exists(filepath):
        logger.warning(f"HTML file not found: {filepath}")
        return (
            "<html><body style='font-family:sans-serif;padding:40px;"
            "background:#0B1120;color:#fff;'>"
            f"<h2 style='color:#B8922A;'>AuditMind</h2>"
            f"<p>Page not found: {filename}</p>"
            "</body></html>"
        )
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def apply_rulebook(df: pd.DataFrame) -> list:
    fired = []
    for _, row in df.iterrows():
        rd = row.to_dict()
        for rule in RULEBOOK:
            try:
                if rule["condition"](rd):
                    fired.append({
                        "invoice_id":  str(rd.get("Invoice_ID", "")),
                        "amount":      round(float(rd.get("Amount", 0)), 2),
                        "rule_id":     rule["id"],
                        "description": rule["description"],
                        "risk_detail": rule["risk"],
                        "severity":    rule["severity"],
                    })
            except Exception as exc:
                logger.warning(f"Rule {rule['id']} failed: {exc}")
    return fired


def anomaly_label(score: float) -> str:
    if score >= 0.70: return "High"
    if score >= 0.50: return "Medium"
    return "Low"


# ============================================================
# FINANCIAL STRESS INDICATOR — v1.0
# ============================================================
FSI_VERSION = "2025.06.v1"

def compute_financial_stress_indicator(df: pd.DataFrame) -> dict:
    score   = 0
    signals = []
    details = {}

    total_spend = float(df["Amount"].sum())
    total_count = len(df)

    if total_count < 5 or total_spend == 0:
        return {
            "fsi_score":    0,
            "fsi_level":    "Insufficient data",
            "fsi_signals":  [],
            "fsi_details":  {},
            "fsi_version":  FSI_VERSION,
            "fsi_disclaimer": "Minimum 5 transactions required for FSI calculation.",
        }

    # Signal 1 — Supplier concentration
    sup_spend = df.groupby("Supplier_Type")["Amount"].sum().sort_values(ascending=False)
    top3_pct  = float(sup_spend.head(3).sum() / total_spend * 100)
    s1 = 20 if top3_pct >= 80 else (12 if top3_pct >= 60 else (6 if top3_pct >= 40 else 0))
    score += s1
    details["supplier_concentration"] = {
        "top3_spend_pct": round(top3_pct, 1),
        "risk_level":     "High" if s1==20 else ("Medium" if s1==12 else ("Low" if s1==6 else "None")),
        "points":         s1,
    }
    if s1 >= 12:
        signals.append(f"Supplier concentration: top 3 supplier types account for {top3_pct:.0f}% of total spend — dependency risk.")

    # Signal 2 — Payment escalation
    if total_count >= 6:
        third     = total_count // 3
        early_avg = float(df.iloc[:third]["Amount"].mean())
        late_avg  = float(df.iloc[-third:]["Amount"].mean())
        esc_ratio = late_avg / early_avg if early_avg > 0 else 1.0
        s2 = 20 if esc_ratio >= 2.0 else (12 if esc_ratio >= 1.5 else (6 if esc_ratio >= 1.2 else 0))
        score += s2
        details["payment_escalation"] = {
            "early_avg":        round(early_avg, 2),
            "late_avg":         round(late_avg, 2),
            "escalation_ratio": round(esc_ratio, 2),
            "risk_level":       "High" if s2==20 else ("Medium" if s2==12 else ("Low" if s2==6 else "None")),
            "points":           s2,
        }
        if s2 >= 12:
            signals.append(f"Payment escalation: average invoice amount has increased {esc_ratio:.1f}x from early to recent transactions.")
    else:
        details["payment_escalation"] = {"risk_level": "Insufficient data", "points": 0}

    # Signal 3 — Round number rate
    rnd_count = int(((df["Amount"] >= 1000) & (df["Amount"] % 1000 == 0)).sum())
    rnd_pct   = float(rnd_count / total_count * 100)
    s3 = 20 if rnd_pct >= 40 else (12 if rnd_pct >= 25 else (6 if rnd_pct >= 15 else 0))
    score += s3
    details["round_number_rate"] = {
        "round_invoices": rnd_count,
        "round_pct":      round(rnd_pct, 1),
        "risk_level":     "High" if s3==20 else ("Medium" if s3==12 else ("Low" if s3==6 else "None")),
        "points":         s3,
    }
    if s3 >= 12:
        signals.append(f"Round number rate: {rnd_pct:.0f}% of invoices are round numbers — possible documentation weakness.")

    # Signal 4 — Duplicate payment rate
    dup_count = int(df.duplicated(subset=["Amount", "Supplier_Type"], keep=False).sum())
    dup_pct   = float(dup_count / total_count * 100)
    s4 = 20 if dup_pct >= 30 else (12 if dup_pct >= 15 else (6 if dup_pct >= 5 else 0))
    score += s4
    details["duplicate_payment_rate"] = {
        "duplicate_count": dup_count,
        "duplicate_pct":   round(dup_pct, 1),
        "risk_level":      "High" if s4==20 else ("Medium" if s4==12 else ("Low" if s4==6 else "None")),
        "points":          s4,
    }
    if s4 >= 12:
        signals.append(f"Duplicate payments: {dup_pct:.0f}% of transactions show duplicate amount-supplier combinations — weak payment controls.")

    # Signal 5 — High value concentration
    top3_inv_pct = float(df["Amount"].nlargest(3).sum() / total_spend * 100)
    s5 = 20 if top3_inv_pct >= 70 else (12 if top3_inv_pct >= 50 else (6 if top3_inv_pct >= 35 else 0))
    score += s5
    details["high_value_concentration"] = {
        "top3_invoice_pct": round(top3_inv_pct, 1),
        "risk_level":       "High" if s5==20 else ("Medium" if s5==12 else ("Low" if s5==6 else "None")),
        "points":           s5,
    }
    if s5 >= 12:
        signals.append(f"High value concentration: top 3 invoices represent {top3_inv_pct:.0f}% of total spend — lumpy cash outflow risk.")

    fsi_level = (
        "Low"      if score <= 25 else
        "Moderate" if score <= 50 else
        "Elevated" if score <= 75 else
        "High"
    )

    logger.info(f"FSI complete | score={score} | level={fsi_level}")

    return {
        "fsi_score":   score,
        "fsi_level":   fsi_level,
        "fsi_signals": signals,
        "fsi_details": details,
        "fsi_version": FSI_VERSION,
        "fsi_disclaimer": (
            "The Financial Stress Indicator is derived from transaction patterns only "
            "and is not a formal insolvency assessment. A complete solvency evaluation "
            "requires full balance sheet and profit and loss data reviewed by a qualified "
            "accountant. All findings must be reviewed by a professional before any action is taken."
        ),
    }


# ============================================================
# CORE ANALYSIS ENGINE
# ============================================================

def analyse_dataframe(df: pd.DataFrame) -> dict:
    required = ["Amount", "Invoice_ID", "VAT_Code", "Supplier_Type"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Your CSV has: {list(df.columns)}")

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Amount"]).copy()

    if len(df) == 0:
        raise ValueError("No valid numeric values found in the Amount column.")

    df["Invoice_ID"]    = df["Invoice_ID"].astype(str).str.strip()
    df["Supplier_Type"] = df["Supplier_Type"].astype(str).str.strip()

    df["VAT_Code_Original"] = df["VAT_Code"].astype(str).str.strip()
    df["VAT_Code"]          = df["VAT_Code_Original"].apply(normalise_vat)

    avg_amount    = df["Amount"].mean()
    median_amount = df["Amount"].median()

    df["log_amount"]       = np.log1p(df["Amount"])
    df["amount_ratio"]     = df["Amount"] / (avg_amount + 1e-9)
    df["is_round_number"]  = ((df["Amount"] >= 1000) & (df["Amount"] % 1000 == 0)).astype(int)
    df["is_subcontractor"] = df["Supplier_Type"].apply(lambda x: 1 if is_subcontractor(x) else 0)
    df["vat_is_valid"]     = df["VAT_Code"].isin(["Standard", "Reverse", "Zero"]).astype(int)
    df["high_value"]       = (df["Amount"] > 5000).astype(int)

    FEATURES = ["Amount", "log_amount", "amount_ratio", "is_round_number", "is_subcontractor", "vat_is_valid", "high_value"]

    ml_applied = len(df) >= 10
    if ml_applied:
        model = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
        model.fit(df[FEATURES])
        df["anomaly_raw"]      = model.predict(df[FEATURES])
        df["high_anomaly"]     = (df["anomaly_raw"] == -1).astype(int)
        df["confidence_score"] = np.clip(1 - model.decision_function(df[FEATURES]), 0, 1)
        logger.info(f"Isolation Forest | rows={len(df)} | anomalies={df['high_anomaly'].sum()}")
    else:
        logger.info(f"ML skipped — {len(df)} rows (minimum 10)")
        df["high_anomaly"]     = 0
        df["confidence_score"] = 0.0

    df["duplicate"] = df.duplicated(subset=["Amount", "Supplier_Type"], keep=False).astype(int)
    df["vat_issue"] = (~df["VAT_Code"].isin(["Standard", "Reverse", "Zero"])).astype(int)
    df["cis_issue"] = df["Supplier_Type"].apply(lambda x: 1 if is_subcontractor(x) else 0)

    df["risk_score"] = (
        df["high_anomaly"]    * 2 +
        df["duplicate"]       * 2 +
        df["vat_issue"]       * 1 +
        df["cis_issue"]       * 1 +
        df["is_round_number"] * 1 +
        (df["amount_ratio"] > 10).astype(int) * 1
    )

    def classify(s):
        if s >= 4: return "High"
        if s >= 2: return "Medium"
        return "Low"

    df["risk_level"] = df["risk_score"].apply(classify)

    def explain(row):
        r = []
        if row["high_anomaly"]:      r.append("Statistical outlier detected")
        if row["duplicate"]:         r.append("Possible duplicate transaction")
        if row["vat_issue"]:         r.append("Unrecognised VAT code")
        if row["cis_issue"]:         r.append("Subcontractor — CIS review required")
        if row["is_round_number"]:   r.append("Round number amount")
        if row["amount_ratio"] > 10: r.append("Extreme value vs dataset average")
        return ", ".join(r) if r else "No flags — within normal parameters"

    def recommend(row):
        a = []
        if row["duplicate"]:
            a.append("Cross-reference invoice references before approving payment")
        if row["vat_issue"]:
            a.append("Verify VAT treatment with your accountant")
        if row["cis_issue"] and float(row["Amount"]) > 5000:
            a.append("Verify CIS registration at gov.uk before payment")
        elif row["cis_issue"]:
            a.append("Confirm subcontractor is CIS registered with HMRC")
        if row["high_anomaly"]:
            a.append("Request invoice breakdown and confirm against signed contract")
        if row["is_round_number"] and float(row["Amount"]) >= 10000:
            a.append("Request itemised invoice — large round number requires verification")
        if row["amount_ratio"] > 10:
            a.append("Escalate for senior review — value exceeds 10x dataset average")
        return " | ".join(a) if a else "No specific action required"

    df["explanation"] = df.apply(explain, axis=1)
    df["action"]      = df.apply(recommend, axis=1)

    df = df.sort_values(by=["risk_score", "confidence_score", "Amount"], ascending=[False, False, False])

    results = []
    for _, row in df.iterrows():
        results.append({
            "invoice_id":          str(row["Invoice_ID"]),
            "amount":              round(float(row["Amount"]), 2),
            "risk_level":          row["risk_level"],
            "risk_score":          int(row["risk_score"]),
            "high_anomaly":        "Yes" if row["high_anomaly"]    else "No",
            "duplicate":           "Yes" if row["duplicate"]       else "No",
            "vat_issue":           "Yes" if row["vat_issue"]       else "No",
            "cis_issue":           "Yes" if row["cis_issue"]       else "No",
            "round_number":        "Yes" if row["is_round_number"] else "No",
            "amount_ratio":        round(float(row["amount_ratio"]), 2),
            "confidence_score":    round(float(row["confidence_score"]), 4),
            "confidence_level":    anomaly_label(float(row["confidence_score"])),
            "vat_code_original":   str(row.get("VAT_Code_Original", "")),
            "vat_code_normalised": str(row["VAT_Code"]),
            "explanation":         row["explanation"],
            "action":              row["action"],
            "model_status":        "scored" if ml_applied else "skipped",
        })

    flagged = [r for r in results if r["risk_score"] > 0]

    summary = {
        "total":            len(results),
        "flagged":          len(flagged),
        "high_risk":        sum(1 for r in results if r["risk_level"]       == "High"),
        "medium_risk":      sum(1 for r in results if r["risk_level"]       == "Medium"),
        "low_risk":         sum(1 for r in results if r["risk_level"]       == "Low"),
        "high_anomalies":   sum(1 for r in results if r["high_anomaly"]     == "Yes"),
        "duplicates":       sum(1 for r in results if r["duplicate"]        == "Yes"),
        "vat_issues":       sum(1 for r in results if r["vat_issue"]        == "Yes"),
        "cis_issues":       sum(1 for r in results if r["cis_issue"]        == "Yes"),
        "round_numbers":    sum(1 for r in results if r["round_number"]     == "Yes"),
        "avg_confidence":   round(float(df["confidence_score"].mean()), 4),
        "high_conf":        sum(1 for r in results if r["confidence_level"] == "High"),
        "medium_conf":      sum(1 for r in results if r["confidence_level"] == "Medium"),
        "low_conf":         sum(1 for r in results if r["confidence_level"] == "Low"),
        "avg_amount":       round(float(avg_amount), 2),
        "median_amount":    round(float(median_amount), 2),
        "ml_applied":       ml_applied,
        "rulebook_version": RULEBOOK_VERSION,
    }

    rules_fired = apply_rulebook(df)
    rs_dict = {}
    for rf in rules_fired:
        rid = rf["rule_id"]
        if rid not in rs_dict:
            rs_dict[rid] = {"rule_id": rid, "description": rf["description"], "severity": rf["severity"], "times_fired": 0}
        rs_dict[rid]["times_fired"] += 1

    fsi = compute_financial_stress_indicator(df)

    logger.info(f"Analysis complete | total={len(results)} | flagged={len(flagged)} | high={summary['high_risk']} | fsi={fsi['fsi_score']} ({fsi['fsi_level']})")

    return {
        "summary":                    summary,
        "results":                    results,
        "flagged":                    flagged,
        "rules_fired":                rules_fired,
        "rules_summary":              list(rs_dict.values()),
        "financial_stress_indicator": fsi,
        "disclaimer": (
            "This report is a decision-support tool only. "
            "All flagged transactions require human review. "
            "AuditMind does not make compliance determinations. "
            "Consult a qualified accountant for definitive advice."
        ),
    }


# ============================================================
# HTML PAGE ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
def serve_home():
    return HTMLResponse(content=read_html("Home.html"))

@app.get("/login", response_class=HTMLResponse)
def serve_login():
    return HTMLResponse(content=read_html("login.html"))

@app.get("/upload", response_class=HTMLResponse)
def serve_upload():
    return HTMLResponse(content=read_html("upload.html"))

@app.get("/results", response_class=HTMLResponse)
def serve_results():
    return HTMLResponse(content=read_html("results.html"))


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/api/health")
@app.head("/api/health")
def health():
    return {
        "status":           "ok",
        "service":          "AuditMind Intelligence — Decision Support API",
        "version":          "3.2.1",
        "rulebook_version": RULEBOOK_VERSION,
        "fsi_version":      FSI_VERSION,
    }


@app.post("/api/login")
async def login(credentials: dict):
    username = str(credentials.get("username", "")).strip().lower()
    password = str(credentials.get("password", ""))

    # Admin fallback — always works
    if username == "admin" and password == ADMIN_PASSWORD:
        logger.info("Admin login successful")
        return {"success": True, "username": "admin", "access_token": "admin", "role": "admin"}

    # Check Supabase for registered users
    try:
        sb = get_supabase()
        result = sb.table("platform_users").select("*").eq("username", username).eq("is_active", True).execute()

        if not result.data:
            logger.warning(f"Login failed — user not found: {username}")
            raise HTTPException(status_code=401, detail="Invalid username or password")

        user = result.data[0]

        if not verify_password(password, user["hashed_password"]):
            logger.warning(f"Login failed — wrong password: {username}")
            raise HTTPException(status_code=401, detail="Invalid username or password")

        logger.info(f"Login successful: {username}")
        return {
            "success":      True,
            "username":     username,
            "access_token": username,
            "role":         user.get("role", "user"),
            "company":      user.get("company_name", ""),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Supabase login error: {exc}")
        raise HTTPException(status_code=401, detail="Invalid username or password")


@app.post("/api/register")
async def register(data: dict):
    username        = str(data.get("username", "")).strip().lower()
    password        = str(data.get("password", ""))
    email           = str(data.get("email", "")).strip().lower()
    first_name      = str(data.get("first_name", "")).strip()
    last_name       = str(data.get("last_name", "")).strip()
    company_name    = str(data.get("company_name", "")).strip()
    companies_house = str(data.get("companies_house", "")).strip()
    company_type    = str(data.get("company_type", "")).strip()

    if not username or len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")
    if not password or len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Please provide a valid email address.")
    if not company_name:
        raise HTTPException(status_code=400, detail="Company name is required.")

    try:
        sb = get_supabase()

        existing = sb.table("platform_users").select("username").eq("username", username).execute()
        if existing.data:
            raise HTTPException(status_code=409, detail="Username already taken. Please choose another.")

        existing_email = sb.table("platform_users").select("email").eq("email", email).execute()
        if existing_email.data:
            raise HTTPException(status_code=409, detail="Email already registered. Please sign in instead.")

        hashed = hash_password(password)

        sb.table("platform_users").insert({
            "username":        username,
            "email":           email,
            "hashed_password": hashed,
            "first_name":      first_name,
            "last_name":       last_name,
            "company_name":    company_name,
            "companies_house": companies_house,
            "company_type":    company_type,
            "role":            "user",
            "is_active":       True,
        }).execute()

        logger.info(f"New user registered: {username} | company: {company_name}")
        return {"success": True, "username": username, "message": "Account created successfully."}

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Registration error: {exc}")
        raise HTTPException(status_code=500, detail="Registration failed. Please try again.")


@app.post("/api/columns")
async def get_columns(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="File is empty.")
        if len(contents) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File exceeds 5MB limit.")
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")), nrows=5)
        return {
            "columns":       list(df.columns),
            "preview":       df.head(3).fillna("").to_dict(orient="records"),
            "total_columns": len(df.columns),
            "row_count":     len(df),
        }
    except HTTPException:
        raise
    except UnicodeDecodeError:
        raise HTTPException(status_code=422, detail="File must be UTF-8 encoded.")
    except Exception as exc:
        logger.error(f"Column read error: {exc}")
        raise HTTPException(status_code=422, detail="Could not read CSV file.")


@app.post("/api/analyse")
async def analyse(
    file:         UploadFile = File(...),
    amount_col:   str = Form(default="Amount"),
    invoice_col:  str = Form(default="Invoice_ID"),
    vat_col:      str = Form(default="VAT_Code"),
    supplier_col: str = Form(default="Supplier_Type"),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="File is empty.")
        if len(contents) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File exceeds 5MB limit.")
    except HTTPException:
        raise

    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except UnicodeDecodeError:
        raise HTTPException(status_code=422, detail="File must be UTF-8 encoded.")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not parse CSV file.")

    rename_map = {}
    if amount_col and amount_col in df.columns and amount_col != "Amount":
        rename_map[amount_col] = "Amount"
    if invoice_col and invoice_col in df.columns and invoice_col != "Invoice_ID":
        rename_map[invoice_col] = "Invoice_ID"
    elif not invoice_col or invoice_col not in df.columns:
        df["Invoice_ID"] = [f"ROW-{i+1}" for i in range(len(df))]
    if vat_col and vat_col in df.columns and vat_col != "VAT_Code":
        rename_map[vat_col] = "VAT_Code"
    elif not vat_col or vat_col not in df.columns:
        df["VAT_Code"] = "Standard"
    if supplier_col and supplier_col in df.columns and supplier_col != "Supplier_Type":
        rename_map[supplier_col] = "Supplier_Type"
    elif not supplier_col or supplier_col not in df.columns:
        df["Supplier_Type"] = "Supplier"
    if rename_map:
        df = df.rename(columns=rename_map)

    try:
        result = analyse_dataframe(df)
        return JSONResponse(content=result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        logger.error(f"Analysis error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Analysis could not be completed. Please check your data and try again.")