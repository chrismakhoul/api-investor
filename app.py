# app.py  ────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Core data-science libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# (pip install yfinance pandas numpy scikit-learn)

app = FastAPI(
    title="Investor Friendly Prediction API",
    version="1.0.0",
)

# ── CORS ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # ← during dev; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────────────────────────────
#  ALGORITHM – PRIDE
#  (This is a shortened version.  Replace the “… your original logic …”
#   sections with the full code you tested earlier.)
# ────────────────────────────────────────────────────────────────────────
def predict_pride(symbol: str) -> dict[str, float | str]:
    # 1) download full history (max ~20 yrs)
    df_close = yf.download(symbol, period="max", progress=False)["Close"]

    # 2) calculate price × volatility product (your steps)
    returns = df_close.pct_change()
    vol = returns.rolling(5).std()
    product = (df_close * vol).dropna().sort_index()

    # 3) line of best fit
    x = product.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = product.values
    model = LinearRegression().fit(x, y)

    # 4) your existing *area between curves* + Markov logic
    #    ----------------------------------------------------------------
    #    Keep the exact code you had.  It should ultimately set:
    #         direction  = "up" | "down"
    #         probability = float, between 0 and 1
    #    ----------------------------------------------------------------
    # ↓↓↓  PLACEHOLDER so the code runs – replace with real logic ↓↓↓
    direction = "up"
    probability = 0.65
    # ↑↑↑  ----------------------------------------------------------------

    return {"direction": direction, "probability": probability}


# ────────────────────────────────────────────────────────────────────────
#  ALGORITHM – GLUTTONY  (same idea, but thresholds / bins differ)
# ────────────────────────────────────────────────────────────────────────
def predict_gluttony(symbol: str) -> dict[str, float | str]:
    df_close = yf.download(symbol, period="max", progress=False)["Close"]

    # 1) build features
    returns = df_close.pct_change()
    vol = returns.rolling(5).std()
    product = (df_close * vol).dropna().sort_index()

    # 2) linear fit
    x = product.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = product.values
    model = LinearRegression().fit(x, y)

    # 3) your Gluttony-specific Markov logic
    # ↓↓↓  PLACEHOLDER ↓↓↓
    direction = "down"
    probability = 0.58
    # ↑↑↑  Replace with full implementation ↑↑↑

    return {"direction": direction, "probability": probability}


# ────────────────────────────────────────────────────────────────────────
#  ENDPOINT – recomputed on every request (fresh data every day)
# ────────────────────────────────────────────────────────────────────────
@app.get("/predict")
async def predict(ticker: str, algorithm: str = "pride"):
    """
    Query  
        /predict?ticker=AAPL&algorithm=pride

    Returns  
        { "direction": "up", "probability": 0.83 }
    """
    ticker = ticker.upper()

    try:
        match algorithm.lower():
            case "pride":
                return predict_pride(ticker)
            case "gluttony":
                return predict_gluttony(ticker)
            case _:
                raise HTTPException(400, f"Unknown algorithm '{algorithm}'")
    except Exception as exc:
        raise HTTPException(500, f"prediction failed: {exc}")
