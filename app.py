# app.py  ─────────────────────────────────────────────────────────
import os, io, json, math, asyncio, datetime as dt
from functools import lru_cache
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sklearn.linear_model import LinearRegression

# ─────────────────────────────── constants
ROLLING_WINDOW = 5
ALLOWED_ALGOS = {"pride", "gluttony"}

# ─────────────────────────────── FastAPI
app = FastAPI(title="Pride & Gluttony Predictor API",
              description="Predicts next-day ↑ / ↓ direction with probability using two custom algorithms.")

# ─────────────────────────────── helper: fetch full price history
@lru_cache(maxsize=256)
def get_history(ticker: str) -> pd.Series:
    df = yf.download(ticker, period="max", progress=False)
    if df.empty:
        raise ValueError(f"Ticker {ticker!r} not found.")
    return df["Close"].dropna()

# ─────────────────────────────── core logic identical to your code
def price_vol_product(series: pd.Series) -> pd.Series:
    returns = series.pct_change()
    volatility = returns.rolling(window=ROLLING_WINDOW).std()
    return (series * volatility).dropna().sort_index()

def build_markov_prediction(product: pd.Series):
    # === ALL LINES below replicate your algorithm as-is ======================
    x = np.array(product.index.map(pd.Timestamp.toordinal)).reshape(-1, 1)
    y = product.values
    mask = ~np.isnan(y)
    x_clean, y_clean = x[mask], y[mask]

    threshold = 6
    exclude_indices = np.where(y_clean >= threshold)[0]
    mask2 = y_clean < threshold
    x_clean, y_clean = x_clean[mask2], y_clean[mask2]
    x_dates_clean = pd.to_datetime(
        [pd.Timestamp.fromordinal(int(i)) for i in x_clean.flatten()]
    )

    model = LinearRegression().fit(x_clean, y_clean)
    y_pred = model.predict(x_clean)
    difference = y_clean - y_pred
    epsilon = 1e-4 * np.max(np.abs(y_clean))

    signs = np.where(difference > epsilon, 1,
                     np.where(difference < -epsilon, -1, 0))
    zero_cross = np.where(np.diff(signs) != 0)[0]
    seg_idx = np.concatenate(([0], zero_cross + 1, [len(difference)]))

    areas = []
    tot_above = tot_below = 0.0
    for i in range(len(seg_idx) - 1):
        s, e = seg_idx[i], seg_idx[i + 1]
        if e - s < 1:           # need at least 1 pt
            continue
        if np.intersect1d(np.arange(s, e), exclude_indices).size:
            continue

        seg_x = x_clean[s:e].flatten()
        seg_y = y_clean[s:e]
        seg_yhat = y_pred[s:e]

        if len(seg_x) >= 2:
            area = np.trapz(seg_y, seg_x) - np.trapz(seg_yhat, seg_x)
        else:
            area = (seg_y[0] - seg_yhat[0]) * 1

        pos = "Above" if area > epsilon else "Below" if area < -epsilon else "Neutral"
        if pos == "Above":
            tot_above += area
        elif pos == "Below":
            tot_below += area
        areas.append({"Segment": i+1, "Area": area, "Position": pos})

    areas_df = pd.DataFrame(areas)
    areas_df = areas_df[areas_df["Area"] != 0].reset_index(drop=True)

    # === custom binning & Markov chain (unchanged) ==========================
    pos_df = areas_df[areas_df["Area"] > 0].copy()
    neg_df = areas_df[areas_df["Area"] < 0].copy()

    pct = [0.5, 0.85, 1.0]
    if not pos_df.empty:
        pos_bins = np.unique(
            np.concatenate(([pos_df["Area"].min()],
                            pos_df["Area"].quantile(pct).values,
                            [pos_df["Area"].max()]))
        )
        pos_df["State"] = pd.cut(pos_df["Area"], bins=pos_bins,
                                 labels=[f"Bin {i+1}" for i in range(len(pos_bins)-1)],
                                 include_lowest=True)

    if not neg_df.empty:
        neg_df["Abs"] = neg_df["Area"].abs()
        neg_bins_abs = np.unique(
            np.concatenate(([neg_df["Abs"].min()],
                            neg_df["Abs"].quantile(pct).values,
                            [neg_df["Abs"].max()]))
        )
        neg_bins = -neg_bins_abs[::-1]
        neg_df["State"] = pd.cut(neg_df["Area"], bins=neg_bins,
                                 labels=[f"Bin {6-i}" for i in range(len(neg_bins)-1)],
                                 include_lowest=True)
        neg_df.drop(columns=["Abs"], inplace=True)

    df_binned = pd.concat([pos_df, neg_df], ignore_index=True).dropna(subset=["State"])
    df_binned["State"] = df_binned["State"].astype(str)
    df_binned = df_binned.sort_values("Segment")
    state_seq = df_binned["State"].tolist()

    labels = ["Bin 1", "Bin 2", "Bin 3", "Bin 4", "Bin 5", "Bin 6"]
    n = len(labels)
    T = np.zeros((n, n))
    for i in range(len(state_seq) - 1):
        cur = labels.index(state_seq[i])
        nxt = labels.index(state_seq[i+1])
        T[cur, nxt] += 1
    row_sum = T.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.divide(T, row_sum, where=row_sum != 0)

    cur_state = state_seq[-1]
    cur_idx = labels.index(cur_state)
    probs = P[cur_idx]

    # tomorrow = most probable up/down vs current price
    up_prob   = probs[0] + probs[1] + probs[2]   # Bins 1-3 are “above” zones
    down_prob = probs[3] + probs[4] + probs[5]   # Bins 4-6 are “below”
    if up_prob == down_prob == 0:        # degenerate
        pred, prob = "neutral", 0.5
    elif up_prob >= down_prob:
        pred, prob = "up", float(up_prob)
    else:
        pred, prob = "down", float(down_prob)

    return pred, round(prob, 4)

# ─────────────────────────────── API cache structure
_cache: dict[tuple[str, str], dict] = {}  # key: (algo, ticker)

def compute_prediction(algo: str, ticker: str) -> dict:
    series = get_history(ticker)
    product = price_vol_product(series)
    pred, prob = build_markov_prediction(product)
    return {
        "ticker": ticker.upper(),
        "algorithm": algo,
        "date": dt.date.today().isoformat(),
        "prediction": pred,
        "probability": prob
    }

async def ensure_cached(algo: str, ticker: str) -> dict:
    key = (algo, ticker.upper())
    today = dt.date.today().isoformat()
    entry = _cache.get(key)
    if entry and entry["date"] == today:
        return entry
    data = await asyncio.to_thread(compute_prediction, algo, ticker)
    _cache[key] = data
    return data

# ─────────────────────────────── REST endpoints
class Prediction(BaseModel):
    ticker: str
    algorithm: Literal["pride", "gluttony"]
    date: str
    prediction: Literal["up", "down", "neutral"]
    probability: float

@app.get("/predict/{algorithm}", response_model=Prediction)
async def predict_endpoint(
    algorithm: str,
    ticker: str = Query(..., min_length=1, max_length=10, regex=r"^[A-Za-z\.\-]+$"),
):
    algorithm = algorithm.lower()
    if algorithm not in ALLOWED_ALGOS:
        raise HTTPException(status_code=404, detail="Unknown algorithm.")
    try:
        result = await ensure_cached(algorithm, ticker)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result

@app.post("/refresh")
async def manual_refresh(ticker: str):
    """Force-refresh both algorithms for a ticker."""
    out = {}
    for algo in ALLOWED_ALGOS:
        out[algo] = await ensure_cached(algo, ticker)
    return out

# ─────────────────────────────── background daily refresh
def schedule_refresh():
    async def refresh_all():
        for (algo, tick), _ in list(_cache.items()):
            try:
                _cache[(algo, tick)] = compute_prediction(algo, tick)
            except Exception as e:
                print("Refresh error for", algo, tick, e)

    scheduler = AsyncIOScheduler(timezone="UTC")
    # 22:10 UTC ≈ 6:10 PM US-ET after market close
    scheduler.add_job(refresh_all, "cron", hour=22, minute=10)
    scheduler.start()

@app.on_event("startup")
async def startup_event():
    schedule_refresh()
