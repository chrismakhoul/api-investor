# app.py  ─────────────────────────────────────────────────────────────
"""
Investor-Friendly Prediction API
--------------------------------
GET /predict?ticker=AAPL&algorithm=pride
   → {"direction": "up", "probability": 0.77}
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = FastAPI(title="Investor Friendly API", version="1.3")

# ── CORS (open during dev, restrict later) ───────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # replace with your Framer URL in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────
#  Helper: signed-area segmentation (NaN-safe)
# ─────────────────────────────────────────────────────────────────────
def segment_areas(
    x: np.ndarray,
    y: np.ndarray,
    y_hat: np.ndarray,
    eps: float,
) -> list[str]:
    """Return state sequence ['up','down', …] with NaNs already removed."""
    diff  = y - y_hat
    signs = np.where(diff > eps, 1, np.where(diff < -eps, -1, 0))
    zc    = np.where(np.diff(signs) != 0)[0]
    idx   = np.concatenate(([0], zc + 1, [len(diff)]))

    seq = []
    for i in range(len(idx) - 1):
        s, e = idx[i], idx[i + 1]
        xs, ys, yp = x[s:e].ravel(), y[s:e], y_hat[s:e]
        if xs.size < 2:
            continue
        area = float(np.trapz(ys - yp, xs))
        if abs(area) < eps:
            continue
        seq.append("up" if area > 0 else "down")
    return seq

# ─────────────────────────────────────────────────────────────────────
#  Core routine shared by both algorithms
# ─────────────────────────────────────────────────────────────────────
def build_features(symbol: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) arrays with all NaNs removed."""
    close = yf.download(symbol, period="max", progress=False)["Close"]

    returns = close.pct_change()
    vol     = returns.rolling(5).std()
    prod    = (close * vol).dropna().sort_index()

    # (optional) cap outliers like original code
    prod = prod[prod < 6]

    # remove any remaining NaNs (rare but safe)
    prod = prod.dropna()
    if prod.empty:
        raise ValueError("Not enough valid data after cleaning")

    x = prod.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = prod.values

    mask = ~np.isnan(y)
    if mask.sum() < 2:
        raise ValueError("Insufficient non-NaN points")

    # ⬇︎  ensure X stays 2-D after masking
    x_final = x[mask].reshape(-1, 1)   # <── added .reshape
    y_final = y[mask]

    return x_final, y_final

# ─────────────────────────────────────────────────────────────────────
#  PRIDE
# ─────────────────────────────────────────────────────────────────────
def predict_pride(symbol: str) -> dict[str, float | str]:
    x, y = build_features(symbol)

    model  = LinearRegression().fit(x, y)
    y_hat  = model.predict(x)
    eps    = 1e-4 * np.max(np.abs(y))
    seq    = segment_areas(x, y, y_hat, eps)

    if not seq:
        return {"direction": "up", "probability": 0.5}

    labels = ["up", "down"]
    P = np.zeros((2, 2))
    for a, b in zip(seq, seq[1:]):
        P[labels.index(a), labels.index(b)] += 1
    row_sum = P.sum(axis=1, keepdims=True)
    P = np.divide(P, row_sum, where=row_sum != 0)

    current = labels.index(seq[-1])
    probs   = P[current]
    direction   = labels[int(np.argmax(probs))]
    probability = float(np.max(probs)) if probs.sum() else 0.5
    return {"direction": direction, "probability": probability}

# ─────────────────────────────────────────────────────────────────────
#  GLUTTONY
# ─────────────────────────────────────────────────────────────────────
def predict_gluttony(symbol: str) -> dict[str, float | str]:
    x, y = build_features(symbol)

    model  = LinearRegression().fit(x, y)
    y_hat  = model.predict(x)
    eps    = 1e-4 * np.max(np.abs(y))
    seq    = segment_areas(x, y, y_hat, eps)

    if not seq:
        return {"direction": "up", "probability": 0.5}

    up_ratio   = seq.count("up") / len(seq)
    direction  = "up" if up_ratio >= 0.5 else "down"
    probability = up_ratio if direction == "up" else 1 - up_ratio
    return {"direction": direction, "probability": probability}

# ─────────────────────────────────────────────────────────────────────
#  Endpoint – fresh computation every call
# ─────────────────────────────────────────────────────────────────────
@app.get("/predict")
async def predict(ticker: str, algorithm: str = "pride"):
    """
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
    except ValueError as ve:
        raise HTTPException(422, str(ve))
    except Exception as exc:
        raise HTTPException(500, f"prediction failed: {exc}")
