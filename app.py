# app.py  ───────────────────────────────────────────────────────────────
"""
Investor-Friendly API
• POST /signup   {email,password}
• POST /login    {email,password}   → cookie
• GET  /predict  ?ticker=…&algorithm=pride|gluttony
• GET  /me       returns current user
"""

from fastapi import FastAPI, HTTPException, Depends, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Field, select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from passlib.hash import bcrypt
from datetime import datetime
import os
from dotenv import load_dotenv
import yfinance as yf, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

# ───── env & DB setup ─────────────────────────────────────────────────
load_dotenv(".env", override=True)
RAW_URL = os.environ["DATABASE_URL"]
DATABASE_URL = (
    RAW_URL
    .replace("postgres://",            "postgresql+asyncpg://")
    .replace("postgresql://",          "postgresql+asyncpg://")
    .replace("postgresql+psycopg2://", "postgresql+asyncpg://")
)

engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def get_session():  # dependency
    async with SessionLocal() as s:
        yield s

# ───── Model ──────────────────────────────────────────────────────────
class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    email: str      = Field(index=True, unique=True)
    password_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

# ───── FastAPI init ───────────────────────────────────────────────────
app = FastAPI(title="Investor Friendly API")


@app.get("/health")
async def health():
    return {"ok": True}


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://framer.com",           # preview iframe
        "http://localhost:8000",
        "https://investorfriendly.fr",  # prod domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COOKIE      = "if_session"
COOKIE_AGE  = 60 * 60 * 24 * 7    # 1 week

def hash_pw(pw):      return bcrypt.hash(pw)
def verify_pw(pw, h): return bcrypt.verify(pw, h)

@app.on_event("startup")
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

# ───── Auth endpoints ────────────────────────────────────────────────
@app.post("/signup")
async def signup(data: dict, s: AsyncSession = Depends(get_session)):
    email = data.get("email","").strip().lower()
    pw    = data.get("password","")
    if not email or not pw:
        raise HTTPException(400,"email and password required")
    if await s.exec(select(User).where(User.email==email)).first():
        raise HTTPException(409,"email already registered")
    s.add(User(email=email, password_hash=hash_pw(pw)))
    await s.commit()
    return {"ok":True}

@app.post("/login")
async def login(data: dict, resp: Response, s: AsyncSession = Depends(get_session)):
    email = data.get("email","").strip().lower()
    pw    = data.get("password","")
    user  = (await s.exec(select(User).where(User.email==email))).first()
    if not user or not verify_pw(pw,user.password_hash):
        raise HTTPException(401,"invalid credentials")
    resp.set_cookie(COOKIE,f"{user.id}:{user.password_hash[:10]}",
                    max_age=COOKIE_AGE, httponly=True, samesite="lax")
    return {"ok":True}

async def current_user(req: Request, s: AsyncSession = Depends(get_session)) -> User | None:
    raw=req.cookies.get(COOKIE)
    if not raw or ":" not in raw: return None
    uid,sig=raw.split(":",1)
    u=(await s.exec(select(User).where(User.id==int(uid)))).first()
    return u if u and u.password_hash.startswith(sig) else None

@app.get("/me")
async def me(u: User|None = Depends(current_user)):
    if not u: raise HTTPException(401)
    return {"email":u.email,"created":u.created_at}

# ───── Prediction helpers (unchanged core logic) ─────────────────────
def _segment(x,y,yp,eps):
    diff=y-yp
    s=np.where(diff>eps,1,np.where(diff<-eps,-1,0))
    z=np.where(np.diff(s)!=0)[0]
    idx=np.concatenate(([0],z+1,[len(diff)]))
    out=[]
    for i in range(len(idx)-1):
        xs=x[idx[i]:idx[i+1]].ravel()
        ys=y[idx[i]:idx[i+1]]
        yb=yp[idx[i]:idx[i+1]]
        if xs.size<2: continue
        area=np.trapz(ys-yb,xs)
        if abs(area)<eps: continue
        out.append("up" if area>0 else "down")
    return out

def _features(sym):
    close=yf.download(sym,period="max",progress=False)["Close"]
    vol=close.pct_change().rolling(5).std()
    prod=(close*vol).dropna().sort_index()
    prod=prod[prod<6]; x=prod.index.map(pd.Timestamp.toordinal).values.reshape(-1,1)
    y=prod.values; m=~np.isnan(y); return x[m],y[m]

def _predict(sym):
    x,y=_features(sym)
    if len(y)<2: raise ValueError("not enough data")
    yh=LinearRegression().fit(x,y).predict(x)
    seq=_segment(x,y,yh,1e-4*np.abs(y).max())
    if not seq: return ("up",0.5)
    P=np.zeros((2,2)); lab=["up","down"]
    for a,b in zip(seq,seq[1:]): P[lab.index(a),lab.index(b)]+=1
    P=P/np.clip(P.sum(1,keepdims=True),1e-9,None)
    p=P[lab.index(seq[-1])]; return (lab[int(p.argmax())],float(p.max()))

@app.get("/predict")
async def predict(ticker:str, algorithm:str="pride"):
    ticker=ticker.upper()
    try:
        dir,prob=_predict(ticker)
        return {"direction":dir,"probability":prob}
    except ValueError as e:
        raise HTTPException(422,str(e))
    except Exception as e:
        raise HTTPException(500,f"prediction failed: {e}")
