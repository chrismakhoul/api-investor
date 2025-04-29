import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Depends, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import bcrypt

# ─────────────────────────────────────────────────────────────────────
#  App & CORS
# ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Investor Friendly API", version="1.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # TODO: lock down in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────
#  Database & Auth setup
# ─────────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")
JWT_SECRET   = os.getenv("JWT_SECRET", "CHANGE_ME")

engine       = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base         = declarative_base()

class User(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    email           = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at      = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

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

# ─────────────────────────────────────────────────────────────────────
#  Feature building & segmentation
# ─────────────────────────────────────────────────────────────────────
def segment_areas(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray, eps: float) -> list[str]:
    """Return sequence of 'up'/'down' segments after removing small deviations."""
    diff  = y - y_hat
    signs = np.where(diff > eps, 1, np.where(diff < -eps, -1, 0))
    changes = np.where(np.diff(signs) != 0)[0]
    boundaries = np.concatenate(([0], changes + 1, [len(diff)]))
    seq = []
    for start, end in zip(boundaries, boundaries[1:]):
        xs = x[start:end].ravel()
        ys = y[start:end]
        yp = y_hat[start:end]
        if len(xs) < 2:
            continue
        area = np.trapz(ys - yp, xs)
        if abs(area) <= eps:
            continue
        seq.append("up" if area > 0 else "down")
    return seq

def build_features(symbol: str) -> tuple[np.ndarray, np.ndarray]:
    """Download max history, compute price*5-day-vol, cap outliers, return (x_days, y)."""
    df = yf.download(symbol, period="max", progress=False)[["Close"]]
    if df.empty:
        raise ValueError(f"No data for symbol {symbol}")
    returns = df["Close"].pct_change()
    vol     = returns.rolling(5).std()
    prod    = (df["Close"] * vol).dropna().sort_index()
    # cap outliers
    cap = 6.0
    prod = prod[prod < cap]
    if prod.empty:
        raise ValueError("Not enough valid data after cleaning")
    # x = days since first date
    days = (prod.index - prod.index[0]).days.astype(float)
    x = days.reshape(-1, 1)
    y = prod.values
    # remove any NaNs
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        raise ValueError("Insufficient non-NaN points")
    return x[mask].reshape(-1,1), y[mask]

# ─────────────────────────────────────────────────────────────────────
#  Pride & Gluttony predictors
# ─────────────────────────────────────────────────────────────────────
def predict_pride(symbol: str) -> dict[str, float]:
    x, y = build_features(symbol)
    model = __import__("sklearn.linear_model").linear_model.LinearRegression().fit(x, y)
    y_hat = model.predict(x)
    # epsilon: at least 1e-6 to avoid zero
    eps = max(1e-6, 1e-4 * np.max(np.abs(y)))
    seq = segment_areas(x, y, y_hat, eps)
    # fallback for too few segments
    if len(seq) < 2:
        fallback = seq[-1] if seq else "up"
        return {"direction": fallback, "probability": 0.5}
    # build 2×2 transition matrix
    labels = ["up", "down"]
    P = np.zeros((2,2), dtype=float)
    for a, b in zip(seq, seq[1:]):
        P[labels.index(a), labels.index(b)] += 1
    row_sums = P.sum(axis=1, keepdims=True)
    P = np.divide(P, row_sums, where=row_sums!=0)
    current = labels.index(seq[-1])
    probs   = P[current]
    idx     = int(np.argmax(probs))
    prob    = float(probs[idx])
    if np.isnan(prob):
        prob = 0.5
    return {"direction": labels[idx], "probability": prob}

def predict_gluttony(symbol: str) -> dict[str, float]:
    x, y = build_features(symbol)
    model = __import__("sklearn.linear_model").linear_model.LinearRegression().fit(x, y)
    y_hat = model.predict(x)
    eps = max(1e-6, 1e-4 * np.max(np.abs(y)))
    seq = segment_areas(x, y, y_hat, eps)
    if len(seq) < 2:
        fallback = seq[-1] if seq else "up"
        return {"direction": fallback, "probability": 0.5}
    up_ratio = seq.count("up") / len(seq)
    direction = "up" if up_ratio >= 0.5 else "down"
    probability = up_ratio if direction == "up" else 1 - up_ratio
    if np.isnan(probability):
        probability = 0.5
    return {"direction": direction, "probability": probability}

# ─────────────────────────────────────────────────────────────────────
#  Prediction endpoint
# ─────────────────────────────────────────────────────────────────────
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
            raise HTTPException(400, f"Unknown algorithm '{algorithm}'")
    except ValueError as ve:
        raise HTTPException(422, str(ve))
    except Exception as exc:
        raise HTTPException(500, f"prediction failed: {exc}")

# ─────────────────────────────────────────────────────────────────────
#  Auth endpoints
# ─────────────────────────────────────────────────────────────────────
class Token(BaseModel):
    access_token: str
    token_type: str

class UserOut(BaseModel):
    id: int
    email: str

@app.post("/signup", response_model=UserOut)
def signup(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(400, "Email already registered")
    hashed = get_password_hash(password)
    user = User(email=email, hashed_password=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@app.post("/token", response_model=Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    expire = datetime.utcnow() + timedelta(days=7)
    token = jwt.encode({"sub": str(user.id), "exp": expire}, JWT_SECRET, algorithm="HS256")
    return {"access_token": token, "token_type": "bearer"}

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = int(payload.get("sub"))
    except (JWTError, TypeError, ValueError):
        raise credentials_exception
    user = db.get(User, user_id)
    if not user:
        raise credentials_exception
    return user

@app.get("/users/me", response_model=UserOut)
def read_users_me(current_user: User = Depends(get_current_user)):
    return {"id": current_user.id, "email": current_user.email}
