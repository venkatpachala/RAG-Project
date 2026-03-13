from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import sqlite3
import json
import time
from datetime import datetime
from typing import Optional
import os

app = FastAPI(title="APIForge Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "history.db")

# ── Database Setup ─────────────────────────────────────────────────────────────
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            method TEXT,
            url TEXT,
            headers TEXT,
            body TEXT,
            status_code INTEGER,
            response_body TEXT,
            response_headers TEXT,
            elapsed_ms REAL,
            label TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            method TEXT,
            url TEXT,
            headers TEXT,
            body TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── Models ─────────────────────────────────────────────────────────────────────
class RequestPayload(BaseModel):
    method: str
    url: str
    headers: Optional[dict] = {}
    body: Optional[dict] = None
    label: Optional[str] = ""

class FavoritePayload(BaseModel):
    label: str
    method: str
    url: str
    headers: Optional[dict] = {}
    body: Optional[dict] = None

# ── Core Proxy Endpoint ────────────────────────────────────────────────────────
@app.post("/forge")
async def forge_request(payload: RequestPayload):
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            req_headers = payload.headers or {}
            if payload.body:
                req_headers["Content-Type"] = "application/json"

            response = await client.request(
                method=payload.method.upper(),
                url=payload.url,
                headers=req_headers,
                json=payload.body if payload.body else None,
            )

        elapsed = round((time.time() - start) * 1000, 2)

        try:
            resp_body = response.json()
        except Exception:
            resp_body = {"raw": response.text}

        resp_headers = dict(response.headers)

        # Save to history
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO history (timestamp, method, url, headers, body, status_code, response_body, response_headers, elapsed_ms, label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            payload.method.upper(),
            payload.url,
            json.dumps(payload.headers),
            json.dumps(payload.body),
            response.status_code,
            json.dumps(resp_body),
            json.dumps(resp_headers),
            elapsed,
            payload.label or ""
        ))
        conn.commit()
        conn.close()

        return {
            "status_code": response.status_code,
            "elapsed_ms": elapsed,
            "response_body": resp_body,
            "response_headers": resp_headers,
        }

    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Could not connect to target URL.")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timed out after 15s.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── History Endpoints ──────────────────────────────────────────────────────────
@app.get("/history")
def get_history(limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        result.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "method": r["method"],
            "url": r["url"],
            "headers": json.loads(r["headers"] or "{}"),
            "body": json.loads(r["body"] or "null"),
            "status_code": r["status_code"],
            "response_body": json.loads(r["response_body"] or "{}"),
            "response_headers": json.loads(r["response_headers"] or "{}"),
            "elapsed_ms": r["elapsed_ms"],
            "label": r["label"],
        })
    return result

@app.delete("/history/{item_id}")
def delete_history(item_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM history WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()
    return {"deleted": item_id}

@app.delete("/history")
def clear_history():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM history")
    conn.commit()
    conn.close()
    return {"cleared": True}


# ── Favorites Endpoints ────────────────────────────────────────────────────────
@app.post("/favorites")
def save_favorite(payload: FavoritePayload):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO favorites (label, method, url, headers, body, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        payload.label,
        payload.method,
        payload.url,
        json.dumps(payload.headers),
        json.dumps(payload.body),
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()
    return {"saved": True}

@app.get("/favorites")
def get_favorites():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM favorites ORDER BY id DESC").fetchall()
    conn.close()
    return [{
        "id": r["id"],
        "label": r["label"],
        "method": r["method"],
        "url": r["url"],
        "headers": json.loads(r["headers"] or "{}"),
        "body": json.loads(r["body"] or "null"),
        "created_at": r["created_at"],
    } for r in rows]

@app.delete("/favorites/{fav_id}")
def delete_favorite(fav_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM favorites WHERE id = ?", (fav_id,))
    conn.commit()
    conn.close()
    return {"deleted": fav_id}

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}
