# app.py  ────────────────────────────────────────────────────────────────
"""
Investor Friendly Prediction API
--------------------------------
GET /predict?ticker=AAPL&algorithm=pride
    → {"direction": "up", "probability": 0.78}
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Data / ML libs ─────────────────────────────────────────────────────
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# (make sure requirements.txt has: fastapi uvicorn[standard] yfinance
#  pandas numpy scikit-learn)

# ── FastAPI init + CORS for Framer preview / prod domain ───────────────
app = FastAPI(title="Investor Friendly API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # During dev; tighten to your Framer domain later
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────────────────────────────────────────────────────
#  PRIDE ALGORITHM  (wrapped into a function, no globals)
# ───────────────────────────────────────────────────────────────────────
def predict_pride(symbol: str) -> dict[str, float | str]:
    """Full PRIDE pipeline – fresh run every call."""
    # 1 ── Download full daily history
    close = yf.download(symbol, period="max", progress=False)["Close"]

    # 2 ── Price × rolling-vol product
    returns = close.pct_change()
    vol = returns.rolling(5).std()
    prod = (close * vol).dropna().sort_index()

    # 3 ── Linear-regression fit
    x = prod.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = prod.values
    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)

    # 4 ── Area-between-curve segmentation (your original code)
    diff = y - y_pred
    eps = 1e-4 * np.max(np.abs(y))
    signs = np.where(diff > eps, 1, np.where(diff < -eps, -1, 0))
    zc = np.where(np.diff(signs) != 0)[0]
    seg_idx = np.concatenate(([0], zc + 1, [len(diff)]))

    areas, seq = [], []
    for i in range(len(seg_idx) - 1):
        s, e = seg_idx[i], seg_idx[i + 1]
        xs, ys, yp = x[s:e].flatten(), y[s:e], y_pred[s:e]
        area = np.trapz(ys - yp, xs) if len(xs) >= 2 else (ys[0] - yp[0])
        if abs(area) < eps:
            continue
        seq.append("up" if area > 0 else "down")
        areas.append(area)

    # 5 ── Markov transition matrix
    labels = ["up", "down"]
    trans = np.zeros((2, 2))
    for a, b in zip(seq, seq[1:]):
        i, j = labels.index(a), labels.index(b)
        trans[i, j] += 1
    row_sum = trans.sum(axis=1, keepdims=True)
    P = np.divide(trans, row_sum, where=row_sum != 0)

    # 6 ── One-step forecast
    current = seq[-1]
    p_next = P[labels.index(current)]
    dir_next = labels[int(np.argmax(p_next))]
    prob = float(np.max(p_next))

    return {"direction": dir_next, "probability": prob}


# ───────────────────────────────────────────────────────────────────────
#  GLUTTONY ALGORITHM  (same pattern, kept 100 % faithful)
# ───────────────────────────────────────────────────────────────────────
def predict_gluttony(symbol: str) -> dict[str, float | str]:
    close = yf.download(symbol, period="max", progress=False)["Close"]
    returns = close.pct_change()
    vol = returns.rolling(5).std()
    prod = (close * vol).dropna().sort_index()

    x = prod.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = prod.values
    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)

    # area segmentation (same as Pride)
    diff = y - y_pred
    eps = 1e-4 * np.max(np.abs(y))
    signs = np.where(diff > eps, 1, np.where(diff < -eps, -1, 0))
    zc = np.where(np.diff(signs) != 0)[0]
    seg_idx = np.concatenate(([0], zc + 1, [len(diff)]))

    seq = []
    for i in range(len(seg_idx) - 1):
        s, e = seg_idx[i], seg_idx[i + 1]
        xs, ys, yp = x[s:e].flatten(), y[s:e], y_pred[s:e]
        area = np.trapz(ys - yp, xs) if len(xs) >= 2 else (ys[0] - yp[0])
        if abs(area) < eps:
            continue
        seq.append("up" if area > 0 else "down")

    # bin counts (Gluttony’s weighting trick)
    up_ratio = seq.count("up") / len(seq) if seq else 0.5
    direction = "up" if up_ratio >= 0.5 else "down"
    probability = up_ratio if direction == "up" else 1 - up_ratio

    return {"direction": direction, "probability": probability}


# ───────────────────────────────────────────────────────────────────────
#  SINGLE PREDICT ENDPOINT
# ───────────────────────────────────────────────────────────────────────
@app.get("/predict")
async def predict(ticker: str, algorithm: str = "pride"):
    """
    Example
      /predict?ticker=NVDA&algorithm=gluttony
    """
    ticker = ticker.upper()
    algo = algorithm.lower()

    try:
        if algo == "pride":
            return predict_pride(ticker)
        elif algo == "gluttony":
            return predict_gluttony(ticker)
        else:
            raise HTTPException(400, f"unknown algorithm '{algorithm}'")
    except Exception as exc:
        raise HTTPException(500, f"prediction failed: {exc}")
