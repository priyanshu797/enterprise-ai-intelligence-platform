"""Authentication endpoints: register, login, refresh, current user."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import User
from ..schemas import Token, TokenRefresh, UserCreate, UserLogin, UserOut
from ..security.auth import (
    create_access_token,
    create_refresh_token,
    decode_access_token,
    get_current_active_user,
    hash_password,
    verify_password,
)
from ..services.usage import log_activity

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=201)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    log_activity(db, owner_id=user.id, action="register", detail=f"email={user.email}")
    return user


@router.post("/login", response_model=Token)
def login(payload: UserLogin, request: Request, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        log_activity(
            db, owner_id=user.id if user else None, action="login",
            detail=f"email={payload.email}", status="failed",
            ip_address=request.client.host if request.client else "",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive account")
    token = Token(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )
    log_activity(
        db, owner_id=user.id, action="login",
        ip_address=request.client.host if request.client else "",
    )
    return token


@router.post("/refresh", response_model=Token)
def refresh(payload: TokenRefresh, db: Session = Depends(get_db)):
    data = decode_access_token(payload.refresh_token)
    if data.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    user = db.get(User, data.get("sub", ""))
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return Token(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_active_user)):
    return user
