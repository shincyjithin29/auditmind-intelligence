# ============================================================
# AuditMind Intelligence Ltd — main.py v3.1.0
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
#   GET  /api/health  -> health check
#   POST /api/login   -> authenticate user
#   POST /api/columns -> read CSV headers for column mapping
#   POST /api/analyse -> run full risk analysis
#
# CHANGES IN v3.1.0:
#   - /debug endpoint removed (security)
#   - CORS restricted to known origins (security)
#   - Traceback no longer returned to users (security)
#   - /api/login now returns access_token field
#   - import traceback kept for server-side logging only
# ============================================================

import os
import io
import re
import traceback
import logging

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

# ── Folder where main.py lives ────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(
    title="AuditMind Intelligence — Decision Support API",
    version="3.1.0",
    docs_url=None,   # Swagger UI disabled in production
    redoc_url=None,  # ReDoc disabled in production
)

# ── CORS — restricted to known origins ───────────────────────
ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "https://auditmind-intelligence.onrender.com",
    "https://www.auditmind.co.uk",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ============================================================
# VALID USERS
# In production move passwords to environment variables.
# Never store plaintext passwords in a production system.
# ============================================================
VALID_USERS = {
    "admin": os.getenv("AUDITMIND_ADMIN_PASSWORD", "secure123")
}

# ============================================================
# FLEXIBLE SUBCONTRACTOR DETECTION
# Handles: Subcontractor, Sub-Contractor, sub contractor,
#          sub_contractor, SUBCONTRACTOR etc.
# ============================================================
def is_subcontractor(value: str) -> bool:
    cleaned = re.sub(r'[\s\-_]', '', str(value).strip().lower())
    return cleaned == 'subcontractor'


# ============================================================
# COMPLIANCE RULEBOOK — versioned
# Version: 2025.06.v1
# To add a new rule append a new dict to this list.
# Each rule has: id, description, condition, risk, severity
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
# Maps accounting system codes to standard values.
# Add new mappings here as new systems are tested.
# ============================================================
VAT_MAP = {
    # Standard rate
    "t1": "Standard", "output2": "Standard", "output": "Standard",
    "20%vat": "Standard", "20% vat": "Standard", "sr": "Standard",
    "standardrated": "Standard", "vat20": "Standard",
    # Reverse charge
    "t2": "Reverse", "rc": "Reverse", "reversecharge": "Reverse",
    "domesticreversecharge": "Reverse",
    # Zero rated
    "t0": "Zero", "z": "Zero", "zerorated": "Zero",
    "zeroratedexpenses": "Zero", "0%": "Zero",
    # Exempt
    "exempt": "Exempt", "e": "Exempt",
    # Input
    "input": "Input",
}

def normalise_vat(code: str) -> str:
    cleaned = re.sub(r'[\s\-_]', '', str(code).strip().lower())
    return VAT_MAP.get(cleaned, str(code).strip())


# ============================================================
# HELPERS
# ============================================================

def read_html(filename: str) -> str:
    """Read an HTML file from the same folder as main.py."""
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
    """Apply every rule in RULEBOOK to every row."""
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
# CORE ANALYSIS ENGINE
# ============================================================

def analyse_dataframe(df: pd.DataFrame) -> dict:
    """
    Run full decision-support risk analysis.
    All flags require human review before action is taken.
    This is a decision-support tool, not an automated compliance system.
    """

    # ── Validate columns ──────────────────────────────────────
    required = ["Amount", "Invoice_ID", "VAT_Code", "Supplier_Type"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Your CSV has: {list(df.columns)}"
        )

    # ── Clean data ────────────────────────────────────────────
    df["Amount"]        = pd.to_numeric(df["Amount"], errors="coerce")
    df                  = df.dropna(subset=["Amount"]).copy()

    if len(df) == 0:
        raise ValueError(
            "No valid numeric values found in the Amount column. "
            "Please check your data and column mapping."
        )

    df["Invoice_ID"]    = df["Invoice_ID"].astype(str).str.strip()
    df["Supplier_Type"] = df["Supplier_Type"].astype(str).str.strip()

    # ── VAT normalisation ─────────────────────────────────────
    df["VAT_Code_Original"] = df["VAT_Code"].astype(str).str.strip()
    df["VAT_Code"]          = df["VAT_Code_Original"].apply(normalise_vat)

    # ── Feature engineering (7 features) ─────────────────────
    avg_amount = df["Amount"].mean()
    median_amount = df["Amount"].median()

    df["log_amount"]       = np.log1p(df["Amount"])
    df["amount_ratio"]     = df["Amount"] / (avg_amount + 1e-9)
    df["is_round_number"]  = (
        (df["Amount"] >= 1000) & (df["Amount"] % 1000 == 0)
    ).astype(int)
    df["is_subcontractor"] = df["Supplier_Type"].apply(
        lambda x: 1 if is_subcontractor(x) else 0
    )
    df["vat_is_valid"] = df["VAT_Code"].isin(
        ["Standard", "Reverse", "Zero"]
    ).astype(int)
    df["high_value"] = (df["Amount"] > 5000).astype(int)

    FEATURES = [
        "Amount", "log_amount", "amount_ratio",
        "is_round_number", "is_subcontractor",
        "vat_is_valid", "high_value",
    ]

    # ── Isolation Forest ─────────────────────────────────────
    # Skip ML if dataset is too small for reliable results
    ml_applied = len(df) >= 10

    if ml_applied:
        model = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
        model.fit(df[FEATURES])
        df["anomaly_raw"]      = model.predict(df[FEATURES])
        df["high_anomaly"]     = (df["anomaly_raw"] == -1).astype(int)
        df["confidence_score"] = np.clip(
            1 - model.decision_function(df[FEATURES]), 0, 1
        )
        logger.info(f"Isolation Forest applied | rows={len(df)} | anomalies={df['high_anomaly'].sum()}")
    else:
        logger.info(f"ML skipped — only {len(df)} rows (minimum 10 required)")
        df["high_anomaly"]     = 0
        df["confidence_score"] = 0.0

    # ── Duplicate detection ───────────────────────────────────
    # Uses Amount + Supplier_Type — catches same payment to same supplier type
    df["duplicate"] = df.duplicated(
        subset=["Amount", "Supplier_Type"], keep=False
    ).astype(int)

    # ── VAT check ─────────────────────────────────────────────
    df["vat_issue"] = (
        ~df["VAT_Code"].isin(["Standard", "Reverse", "Zero"])
    ).astype(int)

    # ── CIS check ─────────────────────────────────────────────
    df["cis_issue"] = df["Supplier_Type"].apply(
        lambda x: 1 if is_subcontractor(x) else 0
    )

    # ── Risk score ────────────────────────────────────────────
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

    # ── Plain-English explanation ─────────────────────────────
    def explain(row):
        reasons = []
        if row["high_anomaly"]:       reasons.append("Statistical outlier detected")
        if row["duplicate"]:          reasons.append("Possible duplicate transaction")
        if row["vat_issue"]:          reasons.append("Unrecognised VAT code")
        if row["cis_issue"]:          reasons.append("Subcontractor — CIS review required")
        if row["is_round_number"]:    reasons.append("Round number amount")
        if row["amount_ratio"] > 10:  reasons.append("Extreme value vs dataset average")
        return ", ".join(reasons) if reasons else "No flags — within normal parameters"

    # ── Recommended actions ───────────────────────────────────
    def recommend(row):
        actions = []
        if row["duplicate"]:
            actions.append("Cross-reference invoice references before approving payment")
        if row["vat_issue"]:
            actions.append("Verify VAT treatment with your accountant")
        if row["cis_issue"] and float(row["Amount"]) > 5000:
            actions.append("Verify CIS registration at gov.uk before payment")
        elif row["cis_issue"]:
            actions.append("Confirm subcontractor is CIS registered with HMRC")
        if row["high_anomaly"]:
            actions.append("Request invoice breakdown and confirm against signed contract")
        if row["is_round_number"] and float(row["Amount"]) >= 10000:
            actions.append("Request itemised invoice — large round number requires verification")
        if row["amount_ratio"] > 10:
            actions.append("Escalate for senior review — value exceeds 10x dataset average")
        return " | ".join(actions) if actions else "No specific action required"

    df["explanation"] = df.apply(explain, axis=1)
    df["action"]      = df.apply(recommend, axis=1)

    # ── Sort highest risk first ───────────────────────────────
    df = df.sort_values(
        by=["risk_score", "confidence_score", "Amount"],
        ascending=[False, False, False]
    )

    # ── Build results ─────────────────────────────────────────
    results = []
    for _, row in df.iterrows():
        results.append({
            "invoice_id":       str(row["Invoice_ID"]),
            "amount":           round(float(row["Amount"]), 2),
            "risk_level":       row["risk_level"],
            "risk_score":       int(row["risk_score"]),
            "high_anomaly":     "Yes" if row["high_anomaly"]    else "No",
            "duplicate":        "Yes" if row["duplicate"]       else "No",
            "vat_issue":        "Yes" if row["vat_issue"]       else "No",
            "cis_issue":        "Yes" if row["cis_issue"]       else "No",
            "round_number":     "Yes" if row["is_round_number"] else "No",
            "amount_ratio":     round(float(row["amount_ratio"]), 2),
            "confidence_score": round(float(row["confidence_score"]), 4),
            "confidence_level": anomaly_label(float(row["confidence_score"])),
            "vat_code_original":  str(row.get("VAT_Code_Original", "")),
            "vat_code_normalised":str(row["VAT_Code"]),
            "explanation":      row["explanation"],
            "action":           row["action"],
            "model_status":     "scored" if ml_applied else "skipped",
        })

    flagged = [r for r in results if r["risk_score"] > 0]

    # ── Summary ───────────────────────────────────────────────
    summary = {
        "total":            len(results),
        "flagged":          len(flagged),
        "high_risk":        sum(1 for r in results if r["risk_level"]        == "High"),
        "medium_risk":      sum(1 for r in results if r["risk_level"]        == "Medium"),
        "low_risk":         sum(1 for r in results if r["risk_level"]        == "Low"),
        "high_anomalies":   sum(1 for r in results if r["high_anomaly"]      == "Yes"),
        "duplicates":       sum(1 for r in results if r["duplicate"]         == "Yes"),
        "vat_issues":       sum(1 for r in results if r["vat_issue"]         == "Yes"),
        "cis_issues":       sum(1 for r in results if r["cis_issue"]         == "Yes"),
        "round_numbers":    sum(1 for r in results if r["round_number"]      == "Yes"),
        "avg_confidence":   round(float(df["confidence_score"].mean()), 4),
        "high_conf":        sum(1 for r in results if r["confidence_level"]  == "High"),
        "medium_conf":      sum(1 for r in results if r["confidence_level"]  == "Medium"),
        "low_conf":         sum(1 for r in results if r["confidence_level"]  == "Low"),
        "avg_amount":       round(float(avg_amount), 2),
        "median_amount":    round(float(median_amount), 2),
        "ml_applied":       ml_applied,
        "rulebook_version": RULEBOOK_VERSION,
    }

    # ── Rulebook ──────────────────────────────────────────────
    rules_fired = apply_rulebook(df)
    rs_dict = {}
    for rf in rules_fired:
        rid = rf["rule_id"]
        if rid not in rs_dict:
            rs_dict[rid] = {
                "rule_id":     rid,
                "description": rf["description"],
                "severity":    rf["severity"],
                "times_fired": 0,
            }
        rs_dict[rid]["times_fired"] += 1

    logger.info(
        f"Analysis complete | total={len(results)} | flagged={len(flagged)} | "
        f"high={summary['high_risk']} | rules_fired={len(rules_fired)}"
    )

    return {
        "summary":       summary,
        "results":       results,
        "flagged":       flagged,
        "rules_fired":   rules_fired,
        "rules_summary": list(rs_dict.values()),
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
def health():
    return {
        "status":           "ok",
        "service":          "AuditMind Intelligence — Decision Support API",
        "version":          "3.1.0",
        "rulebook_version": RULEBOOK_VERSION,
    }


@app.post("/api/login")
async def login(credentials: dict):
    """
    Validate username and password.
    Returns access_token for use in Authorization header.
    """
    username = str(credentials.get("username", "")).strip().lower()
    password = str(credentials.get("password", ""))

    if VALID_USERS.get(username) == password:
        logger.info(f"Login successful: {username}")
        return {
            "success":      True,
            "username":     username,
            "access_token": username,  # simple token — compatible with login.html
        }

    logger.warning(f"Failed login attempt for username: {username!r}")
    raise HTTPException(status_code=401, detail="Invalid username or password")


@app.post("/api/columns")
async def get_columns(file: UploadFile = File(...)):
    """
    Step 1 of column mapping flow.
    Read CSV and return column headers and preview rows.
    """
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
    """
    Step 2 of column mapping flow.
    Upload CSV with column mapping and run full risk analysis.
    All results are for decision-support only — human review required.
    """
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

    # ── Apply column mapping ──────────────────────────────────
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

    # ── Run analysis ──────────────────────────────────────────
    try:
        result = analyse_dataframe(df)
        return JSONResponse(content=result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        # Log full traceback server-side — never expose to user
        logger.error(f"Analysis error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="Analysis could not be completed. Please check your data and try again."
        )