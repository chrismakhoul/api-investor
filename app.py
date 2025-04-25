# app.py  ────────────────────────────────────────────────────────────
"""
Investor-Friendly Prediction API
--------------------------------
GET /predict?ticker=AAPL&algorithm=pride
  → {"direction": "up", "probability": 0.77}

Every request downloads the latest daily bar from Yahoo Finance, rebuilds
features, recomputes the Markov chain, and returns a fresh probability.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Data science libs ────────────────────────────────────────────────
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ── FastAPI & CORS setup ─────────────────────────────────────────────
app = FastAPI(title="Investor Friendly API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten later (e.g. ["https://investorfriendly.fr"])
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────
#  Helper: compute signed areas between y and ŷ, return state sequence
# ─────────────────────────────────────────────────────────────────────
def segment_areas(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray, eps: float):
    """
    Returns a tuple (seq, areas)
      seq   – list like ["up", "down", ...]
      areas – same length, numeric areas (+ above, – below)
    Ensures all arrays are 1-D and skips segments shorter than 2 pts.
    """
    diff   = y - y_hat
    signs  = np.where(diff > eps, 1, np.where(diff < -eps, -1, 0))
    zc     = np.where(np.diff(signs) != 0)[0]                # zero crossings
    idx    = np.concatenate(([0], zc + 1, [len(diff)]))

    seq, areas = [], []
    for i in range(len(idx) - 1):
        s, e   = idx[i], idx[i + 1]
        xs     = x[s:e].ravel()                              # 1-D
        ys     = y[s:e]
        y_pred = y_hat[s:e]
        if xs.size < 2:
            continue
        area = float(np.trapz(ys - y_pred, xs))
        if abs(area) < eps:
            continue
        seq.append("up" if area > 0 else "down")
        areas.append(area)
    return seq, areas

# ─────────────────────────────────────────────────────────────────────
#  PRIDE algorithm (faithful to your original code)
# ─────────────────────────────────────────────────────────────────────
def predict_pride(symbol: str) -> dict[str, float | str]:
    close = yf.download(symbol, period="max", progress=False)["Close"]
    returns = close.pct_change()
    vol     = returns.rolling(5).std()
    prod    = (close * vol).dropna().sort_index()

    x = prod.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = prod.values
    model  = LinearRegression().fit(x, y)
    y_pred = model.predict(x)

    eps   = 1e-4 * np.max(np.abs(y))
    seq, _ = segment_areas(x, y, y_pred, eps)

    # 1st-order Markov matrix
    labels = ["up", "down"]
    P = np.zeros((2, 2))
    for a, b in zip(seq, seq[1:]):
        P[labels.index(a), labels.index(b)] += 1
    row_sums = P.sum(axis=1, keepdims=True)
    P = np.divide(P, row_sums, where=row_sums != 0)

    current_idx = labels.index(seq[-1]) if seq else 0
    probs_next  = P[current_idx]
    direction   = labels[int(np.argmax(probs_next))]
    probability = float(np.max(probs_next)) if probs_next.sum() else 0.5

    return {"direction": direction, "probability": probability}

# ─────────────────────────────────────────────────────────────────────
#  GLUTTONY algorithm (also flattened & safe)
# ─────────────────────────────────────────────────────────────────────
def predict_gluttony(symbol: str) -> dict[str, float | str]:
    close = yf.download(symbol, period="max", progress=False)["Close"]
    returns = close.pct_change()
    vol     = returns.rolling(5).std()
    prod    = (close * vol).dropna().sort_index()

    x = prod.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = prod.values
    model  = LinearRegression().fit(x, y)
    y_pred = model.predict(x)

    eps   = 1e-4 * np.max(np.abs(y))
    seq, _ = segment_areas(x, y, y_pred, eps)

    up_ratio = seq.count("up") / len(seq) if seq else 0.5
    direction   = "up" if up_ratio >= 0.5 else "down"
    probability = up_ratio if direction == "up" else 1 - up_ratio

    return {"direction": direction, "probability": probability}

# ─────────────────────────────────────────────────────────────────────
#  Endpoint
# ─────────────────────────────────────────────────────────────────────
@app.get("/predict")
async def predict(ticker: str, algorithm: str = "pride"):
    """
    /predict?ticker=NVDA&algorithm=gluttony
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
