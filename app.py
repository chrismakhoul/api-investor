# app.py  ───────────────────────────────────────────────────────────────
"""
Investor-Friendly API, v2
-------------------------
• POST /signup   {email, password}
• POST /login    {email, password}   → sets cookie
• GET  /predict  ?ticker=...&algorithm=pride|gluttony   (unchanged)
• GET  /me       (example protected route)
"""
from fastapi import FastAPI, HTTPException, Depends, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Field, select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from passlib.hash import bcrypt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv(".env", override=True)  # local only

# ──────────────────────────────────  DB  ──────────────────────────────
DATABASE_URL = os.environ["DATABASE_URL"].replace(
    "postgres://", "postgresql+asyncpg://"
)
engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def get_session() -> AsyncSession:
    async with SessionLocal() as session:
        yield session

# ───────────────────────────── Models ─────────────────────────────────
class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    password_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

# ──────────────────────────── FastAPI ────────────────────────────────
app = FastAPI(title="Investor Friendly API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",      # Framer preview
        "https://investorfriendly.fr" # your prod domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COOKIE_NAME     = "if_session"
COOKIE_MAX_AGE  = 60 * 60 * 24 * 7      # 1 week

def hash_pw(pw: str) -> str:      return bcrypt.hash(pw)
def verify_pw(pw, h) -> bool:     return bcrypt.verify(pw, h)

# ──────────────────────────── DB init ────────────────────────────────
@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

# ─────────────────────────── Signup  ─────────────────────────────────
@app.post("/signup")
async def signup(payload: dict, session: AsyncSession = Depends(get_session)):
    email = payload.get("email", "").strip().lower()
    pwd   = payload.get("password", "")
    if not email or not pwd:
        raise HTTPException(400, "email and password required")
    # e-mail already exists?
    if await session.exec(select(User).where(User.email == email)).first():
        raise HTTPException(409, "email already registered")
    user = User(email=email, password_hash=hash_pw(pwd))
    session.add(user)
    await session.commit()
    return {"ok": True, "msg": "account created"}

# ─────────────────────────── Login  ──────────────────────────────────
@app.post("/login")
async def login(payload: dict, response: Response,
                session: AsyncSession = Depends(get_session)):
    email = payload.get("email", "").strip().lower()
    pwd   = payload.get("password", "")
    user = (await session.exec(select(User).where(User.email == email))).first()
    if not user or not verify_pw(pwd, user.password_hash):
        raise HTTPException(401, "invalid credentials")

    cookie_val = f"{user.id}:{user.password_hash[:10]}"
    response.set_cookie(
        COOKIE_NAME,
        cookie_val,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=False,        # set True if you force HTTPS
    )
    return {"ok": True}

# ─────────────────── helper: current user from cookie ────────────────
async def current_user(request: Request,
                       session: AsyncSession = Depends(get_session)) -> User | None:
    raw = request.cookies.get(COOKIE_NAME)
    if not raw or ":" not in raw:
        return None
    uid, sig = raw.split(":", 1)
    user = (await session.exec(select(User).where(User.id == int(uid)))).first()
    if user and user.password_hash.startswith(sig):
        return user
    return None

@app.get("/me")
async def me(user: User = Depends(current_user)):
    if not user:
        raise HTTPException(401)
    return {"email": user.email, "created_at": user.created_at}

# ─────────────────────────  PREDICTION CODE  ─────────────────────────
# (unchanged from the last NaN-safe version; collapsed for brevity)
import yfinance as yf, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

def _segment(x,y,yh,eps):
    diff = y - yh
    s = np.where(diff>eps,1,np.where(diff<-eps,-1,0))
    zc = np.where(np.diff(s)!=0)[0]
    idx = np.concatenate(([0],zc+1,[len(diff)]))
    seq=[]
    for i in range(len(idx)-1):
        xs = x[idx[i]:idx[i+1]].ravel()
        ys = y[idx[i]:idx[i+1]]
        yp = yh[idx[i]:idx[i+1]]
        if xs.size<2: continue
        area=np.trapz(ys-yp,xs)
        if abs(area)<eps: continue
        seq.append("up" if area>0 else "down")
    return seq

def _features(sym):
    close=yf.download(sym,period="max",progress=False)["Close"]
    vol=close.pct_change().rolling(5).std()
    prod=(close*vol).dropna().sort_index()
    prod=prod[prod<6].dropna()
    x=prod.index.map(pd.Timestamp.toordinal).values.reshape(-1,1)
    y=prod.values
    msk=~np.isnan(y)
    if msk.sum()<2: raise ValueError("not enough data")
    return x[msk].reshape(-1,1), y[msk]

def predict_pride(sym):
    x,y=_features(sym)
    yh=LinearRegression().fit(x,y).predict(x)
    seq=_segment(x,y,yh,1e-4*np.max(np.abs(y)))
    if not seq:return{"direction":"up","probability":0.5}
    labels=["up","down"]
    P=np.zeros((2,2))
    for a,b in zip(seq,seq[1:]): P[labels.index(a),labels.index(b)]+=1
    P=P/np.clip(P.sum(1,keepdims=True),1e-9,None)
    p=P[labels.index(seq[-1])]
    return{"direction":labels[int(np.argmax(p))],"probability":float(np.max(p))}

def predict_gluttony(sym):
    x,y=_features(sym)
    yh=LinearRegression().fit(x,y).predict(x)
    seq=_segment(x,y,yh,1e-4*np.max(np.abs(y)))
    if not seq:return{"direction":"up","probability":0.5}
    labels=["up","down"]
    P=np.zeros((2,2))
    for a,b in zip(seq,seq[1:]): P[labels.index(a),labels.index(b)]+=1
    P=P/np.clip(P.sum(1,keepdims=True),1e-9,None)
    p=P[labels.index(seq[-1])]
    return{"direction":labels[int(np.argmax(p))],"probability":float(np.max(p))}

@app.get("/predict")
async def predict(ticker: str, algorithm: str = "pride"):
    ticker = ticker.upper()
    algo   = algorithm.lower()
    try:
        if algo=="pride":   return predict_pride(ticker)
        if algo=="gluttony":return predict_gluttony(ticker)
        raise HTTPException(400,f"unknown algorithm '{algorithm}'")
    except ValueError as ve:
        raise HTTPException(422,str(ve))
    except Exception as exc:
        raise HTTPException(500,f"prediction failed: {exc}")
