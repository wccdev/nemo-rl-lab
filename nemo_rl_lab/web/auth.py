"""SQLite 用户存储 + JWT 认证（团队模式）；本机 --no-auth 跳过。"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer = HTTPBearer(auto_error=False)

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'operator',
    created_at TEXT NOT NULL
);
"""


class UserStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            conn.execute(CREATE_SQL)

    def create_user(self, username: str, password: str, role: str = "operator") -> None:
        h = pwd_ctx.hash(password)
        ts = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute(
                    "INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)",
                    (username, h, role, ts),
                )
            except sqlite3.IntegrityError as e:
                raise ValueError(f"用户已存在: {username}") from e

    def verify(self, username: str, password: str) -> Optional[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if not row or not pwd_ctx.verify(password, row["password_hash"]):
            return None
        return {"id": row["id"], "username": row["username"], "role": row["role"]}

    def count_users(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def make_token(user: dict, secret: str, hours: int) -> str:
    exp = datetime.now(timezone.utc) + timedelta(hours=hours)
    return jwt.encode(
        {"sub": user["username"], "role": user["role"], "exp": exp},
        secret,
        algorithm="HS256",
    )


def decode_token(token: str, secret: str) -> dict:
    try:
        return jwt.decode(token, secret, algorithms=["HS256"])
    except JWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "无效或已过期的登录") from e


class AuthGuard:
    def __init__(self, store: UserStore, secret: str, no_auth: bool):
        self.store = store
        self.secret = secret
        self.no_auth = no_auth

    def current_user(
        self,
        creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> dict:
        if self.no_auth:
            return {"username": "local", "role": "admin"}
        if not creds:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "需要登录")
        payload = decode_token(creds.credentials, self.secret)
        return {"username": payload["sub"], "role": payload.get("role", "operator")}
