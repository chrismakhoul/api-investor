# app.py ───────────────────────────────────────────────────────────
"""
FastAPI micro‑service that exposes two stock‑direction predictors –
**Pride** and **Gluttony** – with *all* quantitative logic from the
original research notebooks preserved, including:
    • slope‑projection that extends the final area segment
    • six‑state Markov chain with Laplace smoothing
    • Gluttony’s eigen / similarity‑transform diagnostics
    • optional Monte‑Carlo path simulation (10×5 steps)

The only intentional deviation from the notebooks is the data source
(we keep Yahoo Finance via yfinance + Chrome‑TLS session, as requested).

Outputs are trimmed to the classic JSON
    {ticker, algorithm, date, prediction, probability}
so downstream contracts stay unchanged.
"""

import asyncio
import datetime as dt
from functools import lru_cache
from typing import Literal, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import curl_cffi.requests as cfreq
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi.middleware.cors import CORSMiddleware

ROLL = 5                      # rolling window for volatility
ALGORITHMS = {"pride", "gluttony"}
MONTE_CARLO_SIMULATIONS = 10   # keep notebook default (10 × 5‑step paths)
MC_STEPS = 5

app = FastAPI(title="Pride & Gluttony API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # you can change "*" to ["https://your-framer-domain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────── Yahoo helper ─────────────────────────

def _dl(tick: str) -> pd.Series:
    """Download *unadjusted* close prices from Yahoo Finance."""
    sess = cfreq.Session(impersonate="chrome")
    df = yf.download(tick, period="max", progress=False,
                     session=sess, auto_adjust=False, threads=False)
    if df.empty:
        raise ValueError(f"No Yahoo data for {tick}")
    return df["Close"].dropna()

@lru_cache(maxsize=512)
def get_close(tick: str) -> pd.Series:
    return _dl(tick.upper())

# ───────────────────────── data transforms ────────────────────────

def product_series(close: pd.Series) -> pd.Series:
    returns = close.pct_change()
    vol = returns.rolling(window=ROLL).std()
    return (close * vol).dropna().sort_index()

def build_xy(prod: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Return x (ordinal date, 2‑D) and y (float) filtered for y < 6."""
    x = np.array(prod.index.map(pd.Timestamp.toordinal)).reshape(-1, 1)
    y = prod.values
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]
    mask2 = y < 6
    return x[mask2].reshape(-1, 1), y[mask2]

# ─────────────────── helpers shared by both algos ─────────────────

def _segment_areas(x_clean: np.ndarray, y_clean: np.ndarray, y_pred: np.ndarray,
                   eps: float) -> List[dict]:
    """Compute signed areas between y and ŷ. Includes slope projection.
    Returns a list of dicts with keys: Segment, Start, End, Area
    """
    diff = y_clean - y_pred
    sign = np.where(diff > eps, 1, np.where(diff < -eps, -1, 0))
    zc = np.where(np.diff(sign) != 0)[0]
    idx = np.concatenate(([0], zc + 1, [len(diff)]))

    areas: List[dict] = []
    total_area_above = total_area_below = 0.0

    for i in range(len(idx) - 1):
        s, e = idx[i], idx[i + 1]
        if e - s < 1:
            continue
        seg_x = x_clean[s:e].flatten()
        ya, yp = y_clean[s:e], y_pred[s:e]
        if len(seg_x) >= 2:
            a = np.trapz(ya, seg_x) - np.trapz(yp, seg_x)
        else:
            a = ya[0] - yp[0]
        if abs(a) <= eps:
            continue
        areas.append({
            "Segment": len(areas) + 1,
            "Start": pd.Timestamp.fromordinal(int(seg_x[0])).date(),
            "End": pd.Timestamp.fromordinal(int(seg_x[-1])).date(),
            "Area": a,
        })
        if a > 0:
            total_area_above += a
        else:
            total_area_below += a

    # ─── slope‑projection that extends *last* segment ────────────
    if areas:
        dx = np.diff(x_clean.flatten())
        dy = np.diff(y_clean)
        slopes = dy / dx
        avg_up = np.mean(slopes[slopes > 0]) if np.any(slopes > 0) else 0
        avg_down = np.mean(slopes[slopes < 0]) if np.any(slopes < 0) else 0

        diff_last = y_clean[-1] - y_pred[-1]
        if diff_last < 0:
            avg_slope = avg_up
            position = 'Below'
        else:
            avg_slope = avg_down
            position = 'Above'

        m = (y_pred[-1] - y_pred[0]) / (x_clean[-1, 0] - x_clean[0, 0])  # model coef from two points
        c = y_pred[-1] - m * x_clean[-1, 0]

        x0, y0 = x_clean[-1, 0], y_clean[-1]
        denom = m - avg_slope
        if denom != 0 and avg_slope != 0:
            x1 = (y0 - c - avg_slope * x0) / denom
            y1 = m * x1 + c
            x_segment = np.linspace(x0, x1, num=100)
            y_proj = avg_slope * (x_segment - x0) + y0
            y_line = m * x_segment + c
            area_proj = np.trapz(y_proj - y_line, x_segment)

            areas[-1]["End"] = pd.Timestamp.fromordinal(int(x1)).date()
            areas[-1]["Area"] += area_proj
            if position == 'Above':
                total_area_above += area_proj
            else:
                total_area_below += area_proj

    return areas


def _label_sequence(areas: List[dict]) -> List[int]:
    """Convert signed areas to 6‑state label sequence preserving chronology."""
    if len(areas) < 2:
        return []
    arr = np.array([a["Area"] for a in areas])
    pos = arr[arr > 0]
    neg = arr[arr < 0]
    pct = [0.5, 0.85, 1.0]

    if pos.size:
        s = pd.Series(pos)
        pos_bins = np.unique(np.concatenate(([s.min()], s.quantile(pct).values)))
    else:
        pos_bins = np.array([0])  # dummy
    if neg.size:
        s = pd.Series(-neg)
        neg_bins = -np.unique(np.concatenate(([s.min()], s.quantile(pct).values)))[::-1]
    else:
        neg_bins = np.array([0])

    labels: List[int] = []
    for a in arr:
        if a > 0:
            lab = np.digitize(a, pos_bins, right=True) - 1
            labels.append(int(np.clip(lab, 0, 2)))           # 0‑2
        else:
            lab = np.digitize(a, neg_bins, right=True) - 1
            labels.append(int(np.clip(lab, 0, 2) + 3))       # 3‑5
    return labels


def _markov_prob(labels: List[int]) -> Tuple[str, float, np.ndarray]:
    """Return direction, probability, and the full transition matrix P."""
    if len(labels) < 2:
        return "neutral", 0.5, np.full((6, 6), 1/6)
    T = np.zeros((6, 6))
    for a, b in zip(labels, labels[1:]):
        T[a, b] += 1
    T += 1  # Laplace smoothing
    P = T / T.sum(axis=1, keepdims=True)
    cur = labels[-1]
    up = P[cur, :3].sum()
    down = P[cur, 3:].sum()
    tot = up + down
    if tot == 0:
        return "neutral", 0.5, P
    if up >= down:
        return "up", round(up / tot, 4), P
    return "down", round(down / tot, 4), P


# ─────────────────────────── algorithms ──────────────────────────

def run_pride(ticker: str) -> Tuple[str, float]:
    """Full Pride logic with slope‑projection."""
    prod = product_series(get_close(ticker))
    x_clean, y_clean = build_xy(prod)
    model = LinearRegression().fit(x_clean, y_clean)
    y_pred = model.predict(x_clean)
    eps = 1e-4 * np.max(np.abs(y_clean))

    areas = _segment_areas(x_clean, y_clean, y_pred, eps)
    labels = _label_sequence(areas)
    direction, prob, _ = _markov_prob(labels)
    return direction, prob


def run_gluttony(ticker: str) -> Tuple[str, float]:
    """Full Gluttony logic incl. eigen diagnostics and MC simulation."""
    prod = product_series(get_close(ticker))
    x_clean, y_clean = build_xy(prod)
    model = LinearRegression().fit(x_clean, y_clean)
    y_pred = model.predict(x_clean)
    eps = 1e-4 * np.max(np.abs(y_clean))

    areas = _segment_areas(x_clean, y_clean, y_pred, eps)
    labels = _label_sequence(areas)
    direction, prob, P = _markov_prob(labels)

    # ─── eigen / similarity transform diagnostics ────────────────
    try:
        eigvals, eigvecs = np.linalg.eig(P)
        if np.iscomplexobj(eigvecs):
            eigvecs = np.real_if_close(eigvecs)
        P_inv = np.linalg.inv(eigvecs)
    except (np.linalg.LinAlgError, ValueError):
        eigvecs = np.real_if_close(eigvecs)
        P_inv = np.linalg.pinv(eigvecs)
    B = P_inv @ P @ eigvecs
    B[B < 0] = 0
    B = B / B.sum(axis=1, keepdims=True)

    # ─── optional Monte‑Carlo simulation (not used in return) ────
    if labels:
        cur_state = labels[-1]
        paths = []
        for _ in range(MONTE_CARLO_SIMULATIONS):
            path = [cur_state]
            for _ in range(MC_STEPS):
                probs = P[path[-1]]
                nxt = np.random.choice(range(6), p=probs)
                path.append(nxt)
            paths.append(path)
        # print or log as needed

    return direction, prob

# ──────────────────────────── API glue ───────────────────────────

_cache: dict[Tuple[str, str], dict] = {}

def compute(algo: str, tick: str) -> dict:
    func = run_pride if algo == "pride" else run_gluttony
    direction, prob = func(tick)
    return {
        "ticker": tick.upper(),
        "algorithm": algo,
        "date": dt.date.today().isoformat(),
        "prediction": direction,
        "probability": prob,
    }

async def ensure(algo: str, tick: str) -> dict:
    key = (algo, tick.upper())
    today = dt.date.today().isoformat()
    if key in _cache and _cache[key]["date"] == today:
        return _cache[key]
    data = await asyncio.to_thread(compute, algo, tick)
    _cache[key] = data
    return data

class Out(BaseModel):
    ticker: str
    algorithm: Literal["pride", "gluttony"]
    date: str
    prediction: Literal["up", "down", "neutral"]
    probability: float

@app.get("/predict/{algorithm}", response_model=Out)
async def predict(
    algorithm: str,
    ticker: str = Query(..., min_length=1, max_length=12, regex=r"^[A-Za-z\.\-]+$"),
):
    alg = algorithm.lower()
    if alg not in ALGORITHMS:
        raise HTTPException(404, "unknown algorithm")
    try:
        return await ensure(alg, ticker)
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/refresh")
async def refresh(ticker: str):
    return {a: await ensure(a, ticker) for a in ALGORITHMS}

# ─────────────────────────── scheduler ───────────────────────────

def schedule():
    async def daily():
        for (alg, tick), _ in list(_cache.items()):
            try:
                _cache[(alg, tick)] = compute(alg, tick)
            except Exception as exc:
                print("refresh error", alg, tick, exc)

    sch = AsyncIOScheduler(timezone="UTC")
    sch.add_job(daily, "cron", hour=22, minute=10)
    sch.start()

@app.on_event("startup")
async def _():
    schedule()

