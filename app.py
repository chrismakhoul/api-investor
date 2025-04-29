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

# ─────────────────────────────────────────────────────────────────────
#  Authentication: Signup, Login & “/users/me”
#  (appended – do NOT change anything above!)
# ─────────────────────────────────────────────────────────────────────

import os
from datetime import datetime, timedelta

from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import bcrypt
from jose import JWTError, jwt
from fastapi import Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# Load your env vars (Render will inject DATABASE_URL & JWT_SECRET)
DATABASE_URL = os.getenv("DATABASE_URL")
JWT_SECRET   = os.getenv("JWT_SECRET")

# SQLAlchemy setup
engine       = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base         = declarative_base()

class User(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    email           = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at      = Column(DateTime, default=datetime.utcnow)

# Create the users table if it doesn't exist
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

from pydantic import BaseModel, EmailStr

# Pydantic model for signup request
class SignupRequest(BaseModel):
    email: EmailStr
    password: str

@app.post("/signup")
def signup(request: SignupRequest, db: Session = Depends(get_db)):
    """
    Expects JSON body:
      { "email": "you@example.com", "password": "secret" }
    """
    # extract fields
    email = request.email
    password = request.password

    # check for existing user
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # hash the password and save new user
    hashed = get_password_hash(password)
    user   = User(email=email, hashed_password=hashed)
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
    expire       = datetime.utcnow() + timedelta(days=7)
    access_token = jwt.encode({"sub": str(user.id), "exp": expire}, JWT_SECRET, algorithm="HS256")
    return {"access_token": access_token, "token_type": "bearer"}

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
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

@app.get("/users/me")
def read_users_me(current_user: User = Depends(get_current_user)):
    return {"id": current_user.id, "email": current_user.email}
