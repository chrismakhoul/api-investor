# =====================================================================
# app.py  –  FastAPI micro-service for PRIDE & GLUTTONY stock forecasts
# Rollback version: PRIDE (full)  •  GLUTTONY (tempered-PRIDE)
# ---------------------------------------------------------------------
# Run:  uvicorn app:app --reload --port 8000
# ---------------------------------------------------------------------
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Literal
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Prediction API", version="1.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in production restrict to your Framer domain
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# 1.  Helpers – data
# ---------------------------------------------------------------------
def _load_close(ticker: str) -> pd.Series:
    df = yf.download(ticker, progress=False, auto_adjust=True, threads=False)
    if df.empty or "Close" not in df:
        raise ValueError(f"Could not fetch prices for {ticker}")
    return df["Close"].dropna().sort_index()

def _price_vol_product(close: pd.Series, window: int = 5) -> pd.Series:
    returns = close.pct_change()
    vol = returns.rolling(window).std()
    return (close * vol).dropna()

# ---------------------------------------------------------------------
# 2.  Segmentation with purple projection
# ---------------------------------------------------------------------
def _segment_areas(prod: pd.Series,
                   threshold: float = 6,
                   eps_scale: float = 1e-4) -> pd.DataFrame:
    if prod.empty:
        return pd.DataFrame(columns=["Segment", "Start", "End", "Area", "Position"])

    x_all = prod.index.map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
    y_all = prod.values
    mask = (~np.isnan(y_all)) & (y_all < threshold)
    x = x_all[mask].reshape(-1, 1)   # keep 2-D
    y = y_all[mask]

    model = LinearRegression().fit(x, y)
    y_hat = model.predict(x)
    eps = eps_scale * np.max(np.abs(y))

    diff = y - y_hat
    signs = np.where(diff > eps, 1, np.where(diff < -eps, -1, 0))
    zeros = np.where(np.diff(signs) != 0)[0]
    idx = np.concatenate(([0], zeros + 1, [len(diff)]))

    segs = []
    for k in range(len(idx) - 1):
        s, e = idx[k], idx[k + 1]
        xs = x[s:e].flatten()
        if xs.size < 1:
            continue
        ys, yh = y[s:e], y_hat[s:e]
        area = np.trapz(ys, xs) - np.trapz(yh, xs)
        if np.abs(area) < eps:
            continue
        segs.append(dict(
            Segment=len(segs) + 1,
            Start=pd.Timestamp.fromordinal(int(xs[0])).date(),
            End=pd.Timestamp.fromordinal(int(xs[-1])).date(),
            Area=area,
            Position="Above" if area > 0 else "Below"
        ))

    # purple projection: extend last segment
    if segs:
        dx = np.diff(x.flatten())
        dy = np.diff(y)
        with np.errstate(divide="ignore", invalid="ignore"):
            slopes = dy / dx
        pos_slopes = slopes[slopes > 0]
        neg_slopes = slopes[slopes < 0]
        avg_up = np.mean(pos_slopes) if pos_slopes.size else np.nan
        avg_down = np.mean(neg_slopes) if neg_slopes.size else np.nan

        x0, y0 = x[-1, 0], y[-1]
        m, c = model.coef_[0], model.intercept_
        avg_slope = avg_up if (y0 - (m * x0 + c)) < 0 else avg_down

        if not np.isnan(avg_slope):
            denom = m - avg_slope
            if denom != 0:
                x1 = (y0 - c - avg_slope * x0) / denom
                xs_proj = np.linspace(x0, x1, 100)
                y_proj = avg_slope * (xs_proj - x0) + y0
                y_line = m * xs_proj + c
                add_area = np.trapz(y_proj - y_line, xs_proj)

                segs[-1]["End"] = pd.Timestamp.fromordinal(int(x1)).date()
                segs[-1]["Area"] += add_area

    return pd.DataFrame(segs)

# ---------------------------------------------------------------------
# 3.  First-order 6-bin Markov chain  (same as notebooks)
# ---------------------------------------------------------------------
def _markov_probs(area_df: pd.DataFrame) -> Dict[str, float]:
    labels = ["Bin 1", "Bin 2", "Bin 3", "Bin 4", "Bin 5", "Bin 6"]

    if area_df.empty:
        return {"up": 0.5, "down": 0.5}

    pos = area_df[area_df.Area > 0]["Area"]
    neg = area_df[area_df.Area < 0]["Area"].abs()
    df = area_df.copy()

    if not pos.empty:
        q = pos.quantile([0.5, 0.85, 1.0])
        bins = np.concatenate(([pos.min()], q.values))
        df.loc[df.Area > 0, "State"] = pd.cut(
            pos, bins=bins, labels=labels[:3], include_lowest=True
        ).astype(str)

    if not neg.empty:
        qn = neg.quantile([0.5, 0.85, 1.0])
        bins = np.concatenate(([neg.min()], qn.values))
        df.loc[df.Area < 0, "State"] = pd.cut(
            neg, bins=bins, labels=labels[5:2:-1], include_lowest=True
        ).astype(str)

    df = df.dropna(subset=["State"])
    seq = df.sort_values("Segment")["State"].tolist()
    idx = {l: i for i, l in enumerate(labels)}

    A = np.zeros((6, 6))
    for i in range(len(seq) - 1):
        A[idx[seq[i]], idx[seq[i + 1]]] += 1
    rs = A.sum(axis=1, keepdims=True)
    A = np.divide(A, rs, where=rs != 0)

    cur = idx[seq[-1]]
    p_up = A[cur, :3].sum()
    return {"up": float(p_up), "down": 1 - float(p_up)}

# ---------------------------------------------------------------------
# 4.  Prediction engines
# ---------------------------------------------------------------------
def pride_predict(ticker: str) -> Dict[str, float]:
    prod = _price_vol_product(_load_close(ticker))
    areas = _segment_areas(prod)
    probs = _markov_probs(areas)
    direction = "up" if probs["up"] >= probs["down"] else "down"
    return {"direction": direction, "probability": round(max(probs["up"], probs["down"]), 3)}

def gluttony_predict(ticker: str) -> Dict[str, float]:
    """
    Lightweight Gluttony:
    • re-use PRIDE’s direction
    • pull the probability 20 % toward 50 % to reduce over-confidence
    """
    base = pride_predict(ticker)
    temp = 0.2                           # 0 → identical to PRIDE, 1 → 50/50
    prob = 0.5 + (base["probability"] - 0.5) * (1 - temp)
    return {"direction": base["direction"], "probability": round(prob, 3)}

# ---------------------------------------------------------------------
# 5.  FastAPI endpoint
# ---------------------------------------------------------------------
class Prediction(BaseModel):
    ticker: str
    algorithm: Literal["pride", "gluttony"]
    direction: Literal["up", "down"]
    probability: float

@app.get("/predict", response_model=Prediction)
def predict(
    ticker: str = Query(..., description="Ticker symbol, e.g. AAPL"),
    algorithm: Literal["pride", "gluttony"] = Query("pride")
):
    ticker = ticker.upper().strip()
    try:
        if algorithm == "pride":
            res = pride_predict(ticker)
        else:
            res = gluttony_predict(ticker)
    except Exception as exc:
        raise HTTPException(400, detail=str(exc))

    return {"ticker": ticker, "algorithm": algorithm, **res}
