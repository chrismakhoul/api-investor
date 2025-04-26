# app.py  ──────────────────────────────────────────────────────────────
"""
Investor-Friendly Prediction API
--------------------------------
GET /predict?ticker=NVDA&algorithm=gluttony
  → {"direction": "down", "probability": 0.61}
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Data / ML libs ────────────────────────────────────────────────────
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ── FastAPI init + (open) CORS for dev ───────────────────────────────
app = FastAPI(title="Investor Friendly API", version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────
#  Helper: robust signed-area segmentation
# ─────────────────────────────────────────────────────────────────────
def segment_areas(
    x: np.ndarray,
    y: np.ndarray,
    y_hat: np.ndarray,
    eps: float,
) -> list[str]:
    """Return sequence ['up','down', …] for consecutive signed areas."""
    diff   = y - y_hat
    signs  = np.where(diff > eps, 1, np.where(diff < -eps, -1, 0))
    zc     = np.where(np.diff(signs) != 0)[0]                # zero-crossings
    idx    = np.concatenate(([0], zc + 1, [len(diff)]))

    seq = []
    for i in range(len(idx) - 1):
        s, e = idx[i], idx[i + 1]
        xs   = x[s:e].ravel()                                # 1-D
        ys   = y[s:e]
        yp   = y_hat[s:e]
        if xs.size < 2:
            continue
        area = float(np.trapz(ys - yp, xs))
        if abs(area) < eps:
            continue
        seq.append("up" if area > 0 else "down")
    return seq

# ─────────────────────────────────────────────────────────────────────
#  PRIDE  – full original logic (simplified comments)
# ─────────────────────────────────────────────────────────────────────
def predict_pride(symbol: str) -> dict[str, float | str]:
    close = yf.download(symbol, period="max", progress=False)["Close"]

    # feature: price × rolling volatility
    ret = close.pct_change()
    vol = ret.rolling(5).std()
    prod = (close * vol).dropna().sort_index()

    # exclude extreme spikes (>= 6) like original code
    threshold = 6
    mask = prod < threshold
    prod = prod[mask]

    x = prod.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = prod.values
    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)

    eps  = 1e-4 * np.max(np.abs(y))
    seq  = segment_areas(x, y, y_pred, eps)

    # first-order Markov chain in two states: up / down
    labels = ["up", "down"]
    P = np.zeros((2, 2))
    for a, b in zip(seq, seq[1:]):
        P[labels.index(a), labels.index(b)] += 1
    rowsum = P.sum(axis=1, keepdims=True)
    P = np.divide(P, rowsum, where=rowsum != 0)

    if not seq:                           # back-stop: no segments
        return {"direction": "up", "probability": 0.5}

    current_idx = labels.index(seq[-1])
    probs       = P[current_idx]
    direction   = labels[int(np.argmax(probs))]
    probability = float(np.max(probs)) if probs.sum() else 0.5

    return {"direction": direction, "probability": probability}

# ─────────────────────────────────────────────────────────────────────
#  GLUTTONY – same feature-set, but decision via up-ratio
# ─────────────────────────────────────────────────────────────────────
def predict_gluttony(symbol: str) -> dict[str, float | str]:
    close = yf.download(symbol, period="max", progress=False)["Close"]

    ret = close.pct_change()
    vol = ret.rolling(5).std()
    prod = (close * vol).dropna().sort_index()

    x = prod.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = prod.values
    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)

    eps  = 1e-4 * np.max(np.abs(y))
    seq  = segment_areas(x, y, y_pred, eps)

    if not seq:
        return {"direction": "up", "probability": 0.5}

    up_ratio   = seq.count("up") / len(seq)
    direction  = "up" if up_ratio >= 0.5 else "down"
    probability = up_ratio if direction == "up" else 1 - up_ratio
    return {"direction": direction, "probability": probability}

# ─────────────────────────────────────────────────────────────────────
#  Single endpoint
# ─────────────────────────────────────────────────────────────────────
@app.get("/predict")
async def predict(ticker: str, algorithm: str = "pride"):
    """
    Example:
      /predict?ticker=TSLA&algorithm=gluttony
    """
    ticker = ticker.upper()
    algo   = algorithm.lower()

    try:
        if algo == "pride":
            return predict_pride(ticker)
        elif algo == "gluttony":
            return predict_gluttony(ticker)
        else:
            raise HTTPException(400, f"Unknown algorithm '{algorithm}'")
    except Exception as exc:
        raise HTTPException(500, f"prediction failed: {exc}")
