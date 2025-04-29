from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from datetime import datetime, timedelta
from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import bcrypt
from jose import JWTError, jwt

# ─────────────────────────────────────────────────────────────────────
# FastAPI app setup
# ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Investor Friendly API", version="1.3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # replace with your Framer URL in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

def segment_areas(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray, eps: float) -> list[str]:
    """Return sequence of 'up'/'down' segments with NaNs removed."""
    diff = y - y_hat
    signs = np.where(diff > eps, 1, np.where(diff < -eps, -1, 0))
    changes = np.where(np.diff(signs) != 0)[0]
    indices = np.concatenate(([0], changes + 1, [len(diff)]))

    seq = []
    for start, end in zip(indices[:-1], indices[1:]):
        seg_x = x[start:end].ravel()
        seg_y = y[start:end]
        seg_yhat = y_hat[start:end]
        if seg_x.size < 2:
            continue
        area = float(np.trapz(seg_y - seg_yhat, seg_x))
        if abs(area) < eps:
            continue
        seq.append("up" if area > 0 else "down")
    return seq


def build_features(symbol: str, cap_threshold: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Download data, compute price*volatility, cap outliers, and return (x_days, y) arrays."""
    data = yf.download(symbol, period="max", progress=False)["Close"].dropna()
    returns = data.pct_change().dropna()
    vol = returns.rolling(window=5).std().dropna()
    prod = (data.loc[vol.index] * vol).dropna().sort_index()

    # Optional: cap extreme outliers
    prod = prod[prod < cap_threshold]
    if prod.empty:
        raise ValueError("Not enough valid data after cleaning")

    # Convert dates to days since start for stable integration
    days = (prod.index - prod.index[0]).days.astype(float)
    x = days.reshape(-1, 1)
    y = prod.values

    mask = ~np.isnan(y)
    if mask.sum() < 2:
        raise ValueError("Insufficient non-NaN points")

    return x[mask].reshape(-1, 1), y[mask]


def predict_pride(symbol: str) -> dict[str, float | str]:
    x, y = build_features(symbol)
    model = LinearRegression().fit(x, y)
    y_hat = model.predict(x)
    eps = 1e-4 * np.max(np.abs(y))
    seq = segment_areas(x, y, y_hat, eps)

    # If not enough segments to estimate transition probabilities
    if len(seq) < 2:
        last = seq[-1] if seq else "up"
        return {"direction": last, "probability": 0.5}

    labels = ["up", "down"]
    P = np.zeros((2, 2))
    for a, b in zip(seq, seq[1:]):
        P[labels.index(a), labels.index(b)] += 1

    # Normalize rows safely
    row_sum = P.sum(axis=1, keepdims=True)
    P = np.divide(P, np.where(row_sum != 0, row_sum, 1), where=row_sum != 0)

    current = labels.index(seq[-1])
    probs = P[current]
    if probs.sum() == 0:
        return {"direction": seq[-1], "probability": 0.5}

    idx = int(np.argmax(probs))
    return {"direction": labels[idx], "probability": float(probs[idx])}


def predict_gluttony(symbol: str) -> dict[str, float | str]:
    x, y = build_features(symbol)
    model = LinearRegression().fit(x, y)
    y_hat = model.predict(x)
    eps = 1e-4 * np.max(np.abs(y))
    seq = segment_areas(x, y, y_hat, eps)

    if len(seq) < 2:
        last = seq[-1] if seq else "up"
        return {"direction": last, "probability": 0.5}

    up_ratio = seq.count("up") / len(seq)
    direction = "up" if up_ratio >= 0.5 else "down"
    probability = up_ratio if direction == "up" else 1 - up_ratio
    return {"direction": direction, "probability": probability}


@app.get("/predict")
async def predict(ticker: str, algorithm: str = "pride"):
    ticker = ticker.upper()
    algo = algorithm.lower()
    try:
        if algo == "pride":
            return predict_pride(ticker)
        elif algo == "gluttony":
            return predict_gluttony(ticker)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown algorithm '{algorithm}'")
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}")


# ─────────────────────────────────────────────────────────────────────
# Authentication: Signup, Login & “/users/me” (unchanged)
# ─────────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")
JWT_SECRET = os.getenv("JWT_SECRET")

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

ts_auth = OAuth2PasswordBearer(tokenUrl="token")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())

def authenticate_user(db: Session, email: str, password: str):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

@app.post("/signup")
def signup(email: str, password: str, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed = get_password_hash(password)
    user = User(email=email, hashed_password=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"id": user.id, "email": user.email}

@app.post("/token")
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    expire = datetime.utcnow() + timedelta(days=7)
    access_token = jwt.encode({"sub": str(user.id), "exp": expire}, JWT_SECRET, algorithm="HS256")
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
def read_users_me(current_user: User = Depends(lambda: None)):
    # Implementation unchanged
    return {"id": current_user.id, "email": current_user.email}
