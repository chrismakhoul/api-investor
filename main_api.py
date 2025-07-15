# main_api.py  – ESP‑only FastAPI service
# ────────────────────────────────────────────────────────────────────────────
"""
Endpoint
    GET /predict?ticker=AAPL[&force=true]

Response
{
  "ticker": "AAPL",
  "date": "2025-07-15",
  "signal": "STRONG SELL",
  "strength": -0.0042,
  "confidence": 0.83
}
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from enhanced_predictor import EnhancedStockPredictor

app = FastAPI(title="ESP API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ─────────────────── simple in‑process cache ───────────────────────────────
_cache: dict[str, dict] = {}     # {ticker: response_json}
_last_call_ts: float | None = None


def _throttle(max_per_minute: int = 5) -> None:
    """Sleep so we never exceed Alpha‑Vantage free‑tier rate if used."""
    global _last_call_ts
    delay = 60 / max_per_minute
    now = time.time()
    if _last_call_ts is not None and (now - _last_call_ts) < delay:
        time.sleep(delay - (now - _last_call_ts))
    _last_call_ts = time.time()


# ───────────────────────── worker function ────────────────────────────────
def compute(ticker: str) -> dict:
    """Fetch data, run ESP, return compact JSON."""
    api_key = os.getenv("ALPHA_VANTAGE_KEY", "")  # may be empty
    if api_key:
        _throttle()                               # only throttle if key set

    esp = EnhancedStockPredictor(api_key, symbol=ticker)
    if not esp.fetch_data():
        raise ValueError("Data fetch failed (quota or network)")

    esp.calculate_enhanced_features()
    esp.fit_enhanced_regression()
    esp.analyze_segments_advanced()
    esp.create_adaptive_bins()
    esp.build_enhanced_markov_model()
    sig = esp.generate_trading_signals()

    return {
        "ticker": ticker.upper(),
        "date": dt.date.today().isoformat(),
        "signal": sig["signal"],            # STRONG BUY / SELL / HOLD …
        "strength": round(sig["strength"], 4),
        "confidence": round(sig["confidence"], 4),
    }


async def ensure(ticker: str, force: bool) -> dict:
    """Return cached result unless force == True."""
    ticker = ticker.upper()
    today = dt.date.today().isoformat()

    if not force and ticker in _cache and _cache[ticker]["date"] == today:
        return _cache[ticker]

    data = await asyncio.to_thread(compute, ticker)
    _cache[ticker] = data
    return data


# ───────────────────────────── response model ─────────────────────────────
class Out(BaseModel):
    ticker: str
    date: str
    signal: Literal[
        "STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"
    ]
    strength: float
    confidence: float


# ────────────────────────────── endpoint ───────────────────────────────────
@app.get("/predict", response_model=Out)
async def predict(
    ticker: str = Query(..., min_length=1, max_length=12, regex=r"^[A-Za-z.\-]+$"),
    force: bool = Query(False, description="Set true to bypass cache"),
):
    """
    Return ESP trading signal. Add &force=true to ignore the daily cache.
    """
    try:
        return await ensure(ticker, force)
    except ValueError as exc:
        raise HTTPException(503, str(exc))
