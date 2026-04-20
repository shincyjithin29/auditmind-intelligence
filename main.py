# ============================================================
# AuditMind Intelligence Ltd — main.py
# ============================================================
# LOCAL:   python -m uvicorn main:app --reload --port 8000
# RENDER:  python -m uvicorn main:app --host 0.0.0.0 --port 8000
#
# PAGES:
#   GET  /           -> Home.html
#   GET  /login      -> login.html
#   GET  /upload     -> upload.html
#   GET  /results    -> results.html
#
# API:
#   GET  /api/health    -> health check
#   GET  /debug         -> shows files on server (remove after testing)
#   POST /api/analyse   -> CSV upload and risk analysis
# ============================================================

import os
import io
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# ── Locate folder where main.py lives ────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(title="AuditMind Intelligence API", version="1.0.0")

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
# ============================================================
RULEBOOK = [
    {
        "id": "CIS-001",
        "description": "Subcontractor invoice over 5000",
        "condition": lambda r: (
            str(r.get("Supplier_Type", "")).strip().lower() == "subcontractor"
            and float(r.get("Amount", 0)) > 5000
        ),
        "risk": "CIS Risk — high-value subcontractor payment",
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
        "risk": "Statistical anomaly — amount deviates significantly from peers",
        "severity": "High",
    },
    {
        "id": "ANO-002",
        "description": "High ML confidence anomaly score 0.70 or above",
        "condition": lambda r: float(r.get("confidence_score", 0)) >= 0.70,
        "risk": "High confidence anomaly — model strongly flags this transaction",
        "severity": "High",
    },
]

# ============================================================
# HELPERS
# ============================================================

def read_html(filename: str) -> str:
    """Read an HTML file from the same folder as main.py."""
    filepath = os.path.join(BASE, filename)
    if not os.path.exists(filepath):
        return f"""
        <html><body style="font-family:sans-serif;padding:40px;background:#0B1120;color:#fff;">
        <h2 style="color:#B8922A;">AuditMind — File Not Found</h2>
        <p>Could not find: <code>{filepath}</code></p>
        <p>Files available: {os.listdir(BASE)}</p>
        </body></html>
        """
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


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


def confidence_label(score: float) -> str:
    if score >= 0.70:
        return "High"
    elif score >= 0.50:
        return "Medium"
    return "Low"


# ============================================================
# ANALYSIS ENGINE
# ============================================================

def analyse_dataframe(df: pd.DataFrame) -> dict:

    # Validate columns
    required = ["Amount", "Invoice_ID", "VAT_Code", "Supplier_Type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean
    df["Amount"]        = pd.to_numeric(df["Amount"], errors="coerce")
    df                  = df.dropna(subset=["Amount"]).copy()
    df["Invoice_ID"]    = df["Invoice_ID"].astype(str).str.strip()
    df["VAT_Code"]      = df["VAT_Code"].astype(str).str.strip()
    df["Supplier_Type"] = df["Supplier_Type"].astype(str).str.strip()

    # Feature engineering
    df["log_amount"] = np.log1p(df["Amount"])

    # Isolation Forest
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(df[["Amount", "log_amount"]])
    df["anomaly_raw"]      = model.predict(df[["Amount", "log_amount"]])
    df["high_anomaly"]     = (df["anomaly_raw"] == -1).astype(int)
    df["confidence_score"] = np.clip(
        1 - model.decision_function(df[["Amount", "log_amount"]]), 0, 1
    )

    # Rule checks
    df["duplicate"] = df.duplicated(
        subset=["Invoice_ID", "Amount"], keep=False
    ).astype(int)
    df["vat_issue"] = (
        ~df["VAT_Code"].isin(["Standard", "Reverse", "Zero"])
    ).astype(int)
    df["cis_issue"] = (
        df["Supplier_Type"].str.strip().str.lower() == "subcontractor"
    ).astype(int)

    # Composite risk score
    df["risk_score"] = (
        df["high_anomaly"] * 2 +
        df["duplicate"]    * 2 +
        df["vat_issue"]    * 1 +
        df["cis_issue"]    * 1
    )

    def classify(s):
        if s >= 4: return "High"
        if s >= 2: return "Medium"
        return "Low"

    df["risk_level"] = df["risk_score"].apply(classify)

    def explain(row):
        r = []
        if row["high_anomaly"]: r.append("High anomaly detected")
        if row["duplicate"]:    r.append("Duplicate transaction")
        if row["vat_issue"]:    r.append("VAT inconsistency")
        if row["cis_issue"]:    r.append("CIS risk")
        return ", ".join(r) if r else "No risk"

    def recommend(row):
        a = []
        if row["duplicate"]:    a.append("Check for duplicate payment before approval")
        if row["vat_issue"]:    a.append("Verify VAT code against transaction type")
        if row["cis_issue"]:    a.append("Review subcontractor CIS compliance and deductions")
        if row["high_anomaly"]: a.append("Investigate unusual transaction amount or pattern")
        return " | ".join(a) if a else "No action needed"

    df["explanation"] = df.apply(explain, axis=1)
    df["action"]      = df.apply(recommend, axis=1)

    # Sort highest risk first
    df = df.sort_values(
        by=["risk_score", "confidence_score", "Amount"],
        ascending=[False, False, False]
    )

    # Build results list
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
        "avg_confidence": round(float(df["confidence_score"].mean()), 4),
        "high_conf":      sum(1 for r in results if r["confidence_level"] == "High"),
        "medium_conf":    sum(1 for r in results if r["confidence_level"] == "Medium"),
        "low_conf":       sum(1 for r in results if r["confidence_level"] == "Low"),
    }

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
# Uses HTMLResponse — reads HTML files as text and returns them.
# This is more reliable than FileResponse on cloud servers.
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
# DEBUG ENDPOINT — visit /debug to see what files Render sees
# Remove this after confirming everything works
# ============================================================

@app.get("/debug")
def debug():
    return {
        "base_dir":    BASE,
        "files_found": sorted(os.listdir(BASE))
    }


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/api/health")
def health():
    return {
        "status":  "ok",
        "service": "AuditMind Intelligence API",
        "version": "1.0.0"
    }


@app.post("/api/login")
async def login(credentials: dict):
    username = credentials.get("username", "")
    password = credentials.get("password", "")
    if VALID_USERS.get(username) == password:
        return {"success": True, "username": username}
    raise HTTPException(status_code=401, detail="Invalid username or password")


@app.post("/api/analyse")
async def analyse(file: UploadFile = File(...)):
    """Upload CSV and receive full AI risk analysis."""
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
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {traceback.format_exc()}"
        )