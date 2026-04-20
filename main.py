# ============================================================
# AuditMind Intelligence Ltd — main.py
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
#   GET  /api/health         -> health check
#   POST /api/columns        -> read CSV headers for column mapping
#   POST /api/analyse        -> run full risk analysis
# ============================================================

import os
import io
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# ── Folder where main.py lives ───────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# APP
# ============================================================
app = FastAPI(title="AuditMind Intelligence API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# VALID USERS
# ============================================================
VALID_USERS = {"admin": "secure123"}

# ============================================================
# COMPLIANCE RULEBOOK
# To add a new rule — just append a new dict to this list
# ============================================================
RULEBOOK = [
    {
        "id": "CIS-001",
        "description": "Subcontractor invoice over £5,000",
        "condition": lambda r: (
            str(r.get("Supplier_Type", "")).strip().lower() == "subcontractor"
            and float(r.get("Amount", 0)) > 5000
        ),
        "risk": "CIS Risk — high-value subcontractor payment requires HMRC verification",
        "severity": "High",
    },
    {
        "id": "CIS-002",
        "description": "Any subcontractor payment — CIS registration check required",
        "condition": lambda r: (
            str(r.get("Supplier_Type", "")).strip().lower() == "subcontractor"
        ),
        "risk": "CIS Risk — verify subcontractor registration status with HMRC",
        "severity": "Medium",
    },
    {
        "id": "DUP-001",
        "description": "Duplicate amount and supplier type detected",
        "condition": lambda r: r.get("duplicate", 0) == 1,
        "risk": "Duplicate transaction — same amount from same supplier type, possible double payment",
        "severity": "High",
    },
    {
        "id": "VAT-001",
        "description": "Invalid or unrecognised VAT code",
        "condition": lambda r: r.get("vat_issue", 0) == 1,
        "risk": "VAT inconsistency — code is not Standard, Reverse, or Zero",
        "severity": "Medium",
    },
    {
        "id": "ANO-001",
        "description": "ML anomaly — unusual transaction amount detected",
        "condition": lambda r: r.get("high_anomaly", 0) == 1,
        "risk": "Statistical anomaly — amount deviates significantly from peer transactions",
        "severity": "High",
    },
    {
        "id": "ANO-002",
        "description": "High ML confidence anomaly score (0.70 or above)",
        "condition": lambda r: float(r.get("confidence_score", 0)) >= 0.70,
        "risk": "High confidence anomaly — model strongly flags this transaction for investigation",
        "severity": "High",
    },
    {
        "id": "RND-001",
        "description": "Suspiciously round number amount",
        "condition": lambda r: (
            float(r.get("Amount", 0)) >= 1000
            and float(r.get("Amount", 0)) % 1000 == 0
        ),
        "risk": "Round number alert — fabricated invoices often use round amounts",
        "severity": "Medium",
    },
    {
        "id": "HVL-001",
        "description": "Very high value transaction — above 10x dataset average",
        "condition": lambda r: float(r.get("amount_ratio", 0)) > 10,
        "risk": "Extreme value — transaction is more than 10x the dataset average amount",
        "severity": "High",
    },
]


# ============================================================
# HELPER — read HTML file safely
# ============================================================
def read_html(filename: str) -> str:
    filepath = os.path.join(BASE, filename)
    if not os.path.exists(filepath):
        return f"""<html><body style="font-family:sans-serif;padding:40px;
        background:#0B1120;color:#fff;">
        <h2 style="color:#B8922A;">AuditMind — File Not Found</h2>
        <p>Could not find: <code>{filepath}</code></p>
        <p>Files in directory: {sorted(os.listdir(BASE))}</p>
        </body></html>"""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================
# HELPER — apply rulebook to dataframe
# ============================================================
def apply_rulebook(df: pd.DataFrame):
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
            except Exception:
                pass
    return fired


# ============================================================
# HELPER — confidence label
# ============================================================
def confidence_label(score: float) -> str:
    if score >= 0.70:
        return "High"
    elif score >= 0.50:
        return "Medium"
    return "Low"


# ============================================================
# CORE ANALYSIS ENGINE
# Fixes applied:
#   1. Duplicate detection uses Amount + Supplier_Type (not Invoice_ID)
#   2. 7 features used for Isolation Forest (not just 2)
#   3. Risk score threshold adjusted for better High/Medium split
#   4. Two new rules: RND-001 (round numbers) and HVL-001 (extreme values)
# ============================================================
def analyse_dataframe(df: pd.DataFrame) -> dict:

    # ── Validate required columns ─────────────────────────────
    required = ["Amount", "Invoice_ID", "VAT_Code", "Supplier_Type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Your CSV has: {list(df.columns)}"
        )

    # ── Clean data ────────────────────────────────────────────
    df["Amount"]        = pd.to_numeric(df["Amount"], errors="coerce")
    df                  = df.dropna(subset=["Amount"]).copy()
    df["Invoice_ID"]    = df["Invoice_ID"].astype(str).str.strip()
    df["VAT_Code"]      = df["VAT_Code"].astype(str).str.strip()
    df["Supplier_Type"] = df["Supplier_Type"].astype(str).str.strip()

    # ── Feature engineering (7 features) ─────────────────────
    avg_amount = df["Amount"].mean()

    df["log_amount"]       = np.log1p(df["Amount"])
    df["amount_ratio"]     = df["Amount"] / avg_amount
    df["is_round_number"]  = ((df["Amount"] >= 1000) & (df["Amount"] % 1000 == 0)).astype(int)
    df["is_subcontractor"] = (df["Supplier_Type"].str.strip().str.lower() == "subcontractor").astype(int)
    df["vat_is_valid"]     = df["VAT_Code"].isin(["Standard", "Reverse", "Zero"]).astype(int)
    df["high_value"]       = (df["Amount"] > 5000).astype(int)

    FEATURES = [
        "Amount",
        "log_amount",
        "amount_ratio",
        "is_round_number",
        "is_subcontractor",
        "vat_is_valid",
        "high_value",
    ]

    # ── Isolation Forest with 7 features ─────────────────────
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(df[FEATURES])
    df["anomaly_raw"]      = model.predict(df[FEATURES])
    df["high_anomaly"]     = (df["anomaly_raw"] == -1).astype(int)
    df["confidence_score"] = np.clip(
        1 - model.decision_function(df[FEATURES]), 0, 1
    )

    # ── FIX: Duplicate detection on Amount + Supplier_Type ────
    # Previously used Invoice_ID + Amount which never matched
    # because all Invoice IDs are unique.
    # Now correctly detects same payment amount to same supplier type.
    df["duplicate"] = df.duplicated(
        subset=["Amount", "Supplier_Type"], keep=False
    ).astype(int)

    # ── VAT check ─────────────────────────────────────────────
    df["vat_issue"] = (
        ~df["VAT_Code"].isin(["Standard", "Reverse", "Zero"])
    ).astype(int)

    # ── CIS check ─────────────────────────────────────────────
    df["cis_issue"] = (
        df["Supplier_Type"].str.strip().str.lower() == "subcontractor"
    ).astype(int)

    # ── Composite risk score ──────────────────────────────────
    # Anomaly   = 2 points  (ML detected)
    # Duplicate = 2 points  (payment fraud risk)
    # VAT issue = 1 point   (compliance risk)
    # CIS issue = 1 point   (regulatory risk)
    # Round no  = 1 point   (fabrication signal)
    # High ratio= 1 point   (extreme value)
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

    # ── Plain-English explanations ────────────────────────────
    def explain(row):
        reasons = []
        if row["high_anomaly"]:         reasons.append("High anomaly detected")
        if row["duplicate"]:            reasons.append("Duplicate transaction")
        if row["vat_issue"]:            reasons.append("VAT inconsistency")
        if row["cis_issue"]:            reasons.append("CIS risk")
        if row["is_round_number"]:      reasons.append("Suspicious round number")
        if row["amount_ratio"] > 10:    reasons.append("Extreme value — over 10x average")
        return ", ".join(reasons) if reasons else "No risk"

    # ── Recommended actions ───────────────────────────────────
    def recommend(row):
        actions = []
        if row["duplicate"]:
            actions.append("Check for duplicate payment before approval")
        if row["vat_issue"]:
            actions.append("Verify VAT code against transaction type")
        if row["cis_issue"]:
            actions.append("Review subcontractor CIS compliance and deductions")
        if row["high_anomaly"]:
            actions.append("Investigate unusual transaction amount or pattern")
        if row["is_round_number"] and row["Amount"] >= 10000:
            actions.append("Verify invoice authenticity — large round number detected")
        if row["amount_ratio"] > 10:
            actions.append("Escalate for senior review — value exceeds 10x dataset average")
        return " | ".join(actions) if actions else "No action needed"

    df["explanation"] = df.apply(explain, axis=1)
    df["action"]      = df.apply(recommend, axis=1)

    # ── Sort highest risk first ───────────────────────────────
    df = df.sort_values(
        by=["risk_score", "confidence_score", "Amount"],
        ascending=[False, False, False]
    )

    # ── Build results list ────────────────────────────────────
    results = []
    for _, row in df.iterrows():
        results.append({
            "invoice_id":       str(row["Invoice_ID"]),
            "amount":           round(float(row["Amount"]), 2),
            "risk_level":       row["risk_level"],
            "high_anomaly":     "Yes" if row["high_anomaly"]    else "No",
            "duplicate":        "Yes" if row["duplicate"]       else "No",
            "vat_issue":        "Yes" if row["vat_issue"]       else "No",
            "cis_issue":        "Yes" if row["cis_issue"]       else "No",
            "round_number":     "Yes" if row["is_round_number"] else "No",
            "amount_ratio":     round(float(row["amount_ratio"]), 2),
            "confidence_score": round(float(row["confidence_score"]), 4),
            "confidence_level": confidence_label(float(row["confidence_score"])),
            "explanation":      row["explanation"],
            "action":           row["action"],
        })

    flagged = [r for r in results if r["explanation"] != "No risk"]

    # ── Summary metrics ───────────────────────────────────────
    summary = {
        "total":          len(results),
        "flagged":        len(flagged),
        "high_risk":      sum(1 for r in results if r["risk_level"]       == "High"),
        "medium_risk":    sum(1 for r in results if r["risk_level"]       == "Medium"),
        "low_risk":       sum(1 for r in results if r["risk_level"]       == "Low"),
        "high_anomalies": sum(1 for r in results if r["high_anomaly"]     == "Yes"),
        "duplicates":     sum(1 for r in results if r["duplicate"]        == "Yes"),
        "vat_issues":     sum(1 for r in results if r["vat_issue"]        == "Yes"),
        "cis_issues":     sum(1 for r in results if r["cis_issue"]        == "Yes"),
        "round_numbers":  sum(1 for r in results if r["round_number"]     == "Yes"),
        "avg_confidence": round(float(df["confidence_score"].mean()), 4),
        "high_conf":      sum(1 for r in results if r["confidence_level"] == "High"),
        "medium_conf":    sum(1 for r in results if r["confidence_level"] == "Medium"),
        "low_conf":       sum(1 for r in results if r["confidence_level"] == "Low"),
        "avg_amount":     round(float(avg_amount), 2),
    }

    # ── Apply rulebook ────────────────────────────────────────
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

    return {
        "summary":       summary,
        "results":       results,
        "flagged":       flagged,
        "rules_fired":   rules_fired,
        "rules_summary": list(rs_dict.values()),
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
# DEBUG — shows files Render can see (safe to keep)
# ============================================================
@app.get("/debug")
def debug():
    return {"base_dir": BASE, "files": sorted(os.listdir(BASE))}


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AuditMind Intelligence API", "version": "2.0.0"}


@app.post("/api/login")
async def login(credentials: dict):
    username = credentials.get("username", "")
    password = credentials.get("password", "")
    if VALID_USERS.get(username) == password:
        return {"success": True, "username": username}
    raise HTTPException(status_code=401, detail="Invalid username or password")


# ── NEW: Column mapping endpoint ──────────────────────────────
# Step 1 of the column mapping flow:
# Upload CSV → get back the column headers so the UI
# can show dropdowns for the user to map them.
@app.post("/api/columns")
async def get_columns(file: UploadFile = File(...)):
    """
    Read a CSV file and return its column headers.
    Used by upload.html to show column mapping dropdowns.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")), nrows=5)
        columns = list(df.columns)
        preview = df.head(3).to_dict(orient="records")
        return {
            "columns": columns,
            "preview": preview,
            "total_columns": len(columns),
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read CSV: {str(e)}")


# ── NEW: Analyse with column mapping ─────────────────────────
# Step 2 of the column mapping flow:
# Upload CSV + column mapping → run analysis with renamed columns.
@app.post("/api/analyse")
async def analyse(
    file:          UploadFile = File(...),
    amount_col:    str = Form(default="Amount"),
    invoice_col:   str = Form(default="Invoice_ID"),
    vat_col:       str = Form(default="VAT_Code"),
    supplier_col:  str = Form(default="Supplier_Type"),
):
    """
    Upload a CSV file and receive full AI risk analysis.

    Parameters (all optional — defaults work for standard CSV):
      file         — the CSV file
      amount_col   — which column contains the amount/value
      invoice_col  — which column contains the invoice reference
      vat_col      — which column contains the VAT code
      supplier_col — which column contains the supplier/contractor type
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # ── Apply column mapping ──────────────────────────────
        # Rename user's column names to our standard names
        rename_map = {}

        if amount_col in df.columns and amount_col != "Amount":
            rename_map[amount_col] = "Amount"

        if invoice_col in df.columns and invoice_col != "Invoice_ID":
            rename_map[invoice_col] = "Invoice_ID"
        elif invoice_col not in df.columns:
            # No invoice column found — generate row numbers
            df["Invoice_ID"] = [f"ROW-{i+1}" for i in range(len(df))]

        if vat_col in df.columns and vat_col != "VAT_Code":
            rename_map[vat_col] = "VAT_Code"
        elif vat_col not in df.columns:
            # No VAT column — default to Standard
            df["VAT_Code"] = "Standard"

        if supplier_col in df.columns and supplier_col != "Supplier_Type":
            rename_map[supplier_col] = "Supplier_Type"
        elif supplier_col not in df.columns:
            # No supplier column — default to Supplier
            df["Supplier_Type"] = "Supplier"

        if rename_map:
            df = df.rename(columns=rename_map)

        # ── Run analysis ──────────────────────────────────────
        result = analyse_dataframe(df)
        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {traceback.format_exc()}"
        )