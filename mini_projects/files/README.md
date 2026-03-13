# ⚡ APIForge — Postman Clone

> A "Postman clone" style API testing & exploration tool — built with FastAPI + Streamlit

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)

---

## ✨ Features

- ⚡ **Full HTTP support** — GET, POST, PUT, PATCH, DELETE
- 🔑 **Auth headers** — Bearer token + custom headers support
- 📦 **Request body** — JSON body editor with validation
- 📜 **History** — every request saved, viewable, replayable
- ⭐ **Favorites** — save and load your most-used requests
- 🎨 **Beautiful JSON viewer** — syntax highlighted responses
- 🧪 **Quick test presets** — PokeAPI, JSONPlaceholder, Open-Meteo, CoinGecko
- 📖 **Learn tab** — API concepts cheat sheet built-in
- 🗄️ **SQLite storage** — persistent history across sessions

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/apiforge.git
cd apiforge
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run (Mac/Linux)
```bash
chmod +x start.sh
./start.sh
```

### 3. Run (Windows)
```bash
start.bat
```

### Or run manually (two terminals)

**Terminal 1 — Backend:**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
streamlit run frontend/app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🗂️ Project Structure

```
apiforge/
├── backend/
│   ├── __init__.py
│   └── main.py          # FastAPI app — proxy + history + favorites
├── frontend/
│   └── app.py           # Streamlit UI
├── data/
│   └── history.db       # SQLite (auto-created)
├── requirements.txt
├── start.sh             # Mac/Linux launcher
├── start.bat            # Windows launcher
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Tech | Why |
|-------|------|-----|
| Frontend | Streamlit | Fast, Python-native UI |
| Backend | FastAPI | Async, modern, auto-docs at `/docs` |
| HTTP Client | httpx | Async HTTP with timeout handling |
| Storage | SQLite | Zero-config persistent storage |
| Validation | Pydantic | Type-safe request models |

---

## 📡 Backend API

FastAPI auto-generates docs at `http://localhost:8000/docs`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/forge` | POST | Proxy & execute any HTTP request |
| `/history` | GET | Fetch request history |
| `/history/{id}` | DELETE | Delete one history item |
| `/history` | DELETE | Clear all history |
| `/favorites` | GET | Fetch saved favorites |
| `/favorites` | POST | Save a new favorite |
| `/favorites/{id}` | DELETE | Delete a favorite |
| `/health` | GET | Backend health check |

---

## 💡 Built while learning APIs

This project was built as a hands-on exercise while learning REST API concepts:
- HTTP Methods (GET/POST/PUT/PATCH/DELETE)
- Status codes and their meanings
- Request structure (path, query params, headers, body)
- JSON as the data transfer format
- FastAPI as a backend framework
- Async HTTP with httpx

---

## 📄 License

MIT — free to use, fork, and build on.
