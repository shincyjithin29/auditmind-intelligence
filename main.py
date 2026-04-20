# ============================================================
# AuditMind Intelligence Ltd — main.py (FastAPI Backend)
# ============================================================
# HOW TO RUN:
#   python -m uvicorn main:app --reload --port 8000
#
# ENDPOINTS:
#   GET  /api/health   — check if server is running
#   POST /api/analyse  — upload CSV, get risk analysis back
# ============================================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import io
import traceback

app = FastAPI(title="AuditMind API", version="1.0.0")

# Allow HTML files to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# VALID USERS  — add more users here if needed
# ============================================================
VALID_USERS = {
    "admin": "secure123"
}

# ============================================================
# COMPLIANCE RULEBOOK
# Each rule has an id, description, condition, and severity.
# To add a new rule just append a new dict to this list.
# ============================================================
RULEBOOK = [
    {
        "id": "CIS-001",
        "description": "Subcontractor invoice over £5,000",
        "condition": lambda r: str(r.get("Supplier_Type","")).strip().lower() == "subcontractor"
                               and float(r.get("Amount", 0)) > 5000,
        "risk": "CIS Risk — high-value subcontractor payment",
        "severity": "High",
    },
    {
        "id": "CIS-002",
        "description": "Any subcontractor payment — CIS registration check required",
        "condition": lambda r: str(r.get("Supplier_Type","")).strip().lower() == "subcontractor",
        "risk": "CIS Risk — verify subcontractor registration status with HMRC",
        "severity": "Medium",
    },
    {
        "id": "DUP-001",
        "description": "Duplicate Invoice ID and Amount detected",
        "condition": lambda r: r.get("duplicate", 0) == 1,
        "risk": "Duplicate transaction — possible double payment risk",
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
        "risk": "High confidence anomaly — model is strongly flagging this transaction",
        "severity": "High",
    },
]


def apply_rulebook(df):
    fired = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        for rule in RULEBOOK:
            try:
                if rule["condition"](row_dict):
                    fired.append({
                        "invoice_id":  str(row_dict.get("Invoice_ID", "")),
                        "amount":      round(float(row_dict.get("Amount", 0)), 2),
                        "rule_id":     rule["id"],
                        "description": rule["description"],
                        "risk_detail": rule["risk"],
                        "severity":    rule["severity"],
                    })
            except Exception:
                pass
    return fired


def confidence_label(score):
    if score >= 0.70:
        return "High"
    elif score >= 0.50:
        return "Medium"
    return "Low"


# ============================================================
# CORE ANALYSIS ENGINE
# ============================================================
def analyse_dataframe(df):
    # Validate required columns
    required = ["Amount", "Invoice_ID", "VAT_Code", "Supplier_Type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean data
    df["Amount"]        = pd.to_numeric(df["Amount"], errors="coerce")
    df                  = df.dropna(subset=["Amount"]).copy()
    df["Invoice_ID"]    = df["Invoice_ID"].astype(str).str.strip()
    df["VAT_Code"]      = df["VAT_Code"].astype(str).str.strip()
    df["Supplier_Type"] = df["Supplier_Type"].astype(str).str.strip()

    # Feature engineering
    df["log_amount"] = np.log1p(df["Amount"])

    # Isolation Forest — anomaly detection + confidence score
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(df[["Amount", "log_amount"]])
    df["anomaly_raw"]      = model.predict(df[["Amount", "log_amount"]])
    df["high_anomaly"]     = df["anomaly_raw"].apply(lambda x: 1 if x == -1 else 0)
    df["confidence_score"] = np.clip(1 - model.decision_function(df[["Amount", "log_amount"]]), 0, 1)

    # Rule-based checks
    df["duplicate"] = df.duplicated(subset=["Invoice_ID", "Amount"], keep=False).astype(int)
    df["vat_issue"] = (~df["VAT_Code"].isin(["Standard", "Reverse", "Zero"])).astype(int)
    df["cis_issue"] = (df["Supplier_Type"].str.strip().str.lower() == "subcontractor").astype(int)

    # Composite risk score
    df["risk_score"] = (
        df["high_anomaly"] * 2 +
        df["duplicate"]    * 2 +
        df["vat_issue"]    * 1 +
        df["cis_issue"]    * 1
    )

    def classify(score):
        if score >= 4: return "High"
        if score >= 2: return "Medium"
        return "Low"

    df["risk_level"] = df["risk_score"].apply(classify)

    # Explanations
    def explain(row):
        reasons = []
        if row["high_anomaly"]: reasons.append("High anomaly detected")
        if row["duplicate"]:    reasons.append("Duplicate transaction")
        if row["vat_issue"]:    reasons.append("VAT inconsistency")
        if row["cis_issue"]:    reasons.append("CIS risk")
        return ", ".join(reasons) if reasons else "No risk"

    def recommend(row):
        actions = []
        if row["duplicate"]:    actions.append("Check for duplicate payment before approval")
        if row["vat_issue"]:    actions.append("Verify VAT code against transaction type")
        if row["cis_issue"]:    actions.append("Review subcontractor CIS compliance and deductions")
        if row["high_anomaly"]: actions.append("Investigate unusual transaction amount or pattern")
        return " | ".join(actions) if actions else "No action needed"

    df["explanation"] = df.apply(explain, axis=1)
    df["action"]      = df.apply(recommend, axis=1)

    # Sort — highest risk first
    df = df.sort_values(by=["risk_score", "confidence_score", "Amount"], ascending=[False, False, False])

    # Build results
    results = []
    for _, row in df.iterrows():
        results.append({
            "invoice_id":       str(row["Invoice_ID"]),
            "amount":           round(float(row["Amount"]), 2),
            "risk_level":       row["risk_level"],
            "high_anomaly":     "Yes" if row["high_anomaly"] else "No",
            "duplicate":        "Yes" if row["duplicate"]    else "No",
            "vat_issue":        "Yes" if row["vat_issue"]    else "No",
            "cis_issue":        "Yes" if row["cis_issue"]    else "No",
            "confidence_score": round(float(row["confidence_score"]), 4),
            "confidence_level": confidence_label(float(row["confidence_score"])),
            "explanation":      row["explanation"],
            "action":           row["action"],
        })

    flagged = [r for r in results if r["explanation"] != "No risk"]

    # Summary
    summary = {
        "total":          len(results),
        "flagged":        len(flagged),
        "high_risk":      sum(1 for r in results if r["risk_level"] == "High"),
        "medium_risk":    sum(1 for r in results if r["risk_level"] == "Medium"),
        "low_risk":       sum(1 for r in results if r["risk_level"] == "Low"),
        "high_anomalies": sum(1 for r in results if r["high_anomaly"] == "Yes"),
        "duplicates":     sum(1 for r in results if r["duplicate"]    == "Yes"),
        "vat_issues":     sum(1 for r in results if r["vat_issue"]    == "Yes"),
        "cis_issues":     sum(1 for r in results if r["cis_issue"]    == "Yes"),
        "avg_confidence": round(float(df["confidence_score"].mean()), 4),
        "high_conf":      sum(1 for r in results if r["confidence_level"] == "High"),
        "medium_conf":    sum(1 for r in results if r["confidence_level"] == "Medium"),
        "low_conf":       sum(1 for r in results if r["confidence_level"] == "Low"),
    }

    # Rulebook
    rules_fired = apply_rulebook(df)
    rules_summary_dict = {}
    for rf in rules_fired:
        rid = rf["rule_id"]
        if rid not in rules_summary_dict:
            rules_summary_dict[rid] = {
                "rule_id":     rid,
                "description": rf["description"],
                "severity":    rf["severity"],
                "times_fired": 0,
            }
        rules_summary_dict[rid]["times_fired"] += 1

    return {
        "summary":       summary,
        "results":       results,
        "flagged":       flagged,
        "rules_fired":   rules_fired,
        "rules_summary": list(rules_summary_dict.values()),
    }


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/api/health")
def health():
    """Check if the server is running."""
    return {"status": "ok", "service": "AuditMind Intelligence API", "version": "1.0.0"}


@app.post("/api/login")
async def login(credentials: dict):
    """Validate username and password."""
    username = credentials.get("username", "")
    password = credentials.get("password", "")
    if VALID_USERS.get(username) == password:
        return {"success": True, "username": username}
    raise HTTPException(status_code=401, detail="Invalid username or password")


@app.post("/api/analyse")
async def analyse(file: UploadFile = File(...)):
    """Upload a CSV file and receive full risk analysis."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    try:
        contents = await file.read()
        df       = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        result   = analyse_dataframe(df)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {traceback.format_exc()}")