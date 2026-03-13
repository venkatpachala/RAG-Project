import streamlit as st
import requests
import json
from datetime import datetime

BACKEND = "http://localhost:8000"

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="APIForge",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    
    /* Cards */
    .forge-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    /* Status badges */
    .status-2xx { background:#1a4731; color:#3fb950; padding:3px 10px; border-radius:20px; font-weight:700; font-size:13px; }
    .status-3xx { background:#3d2d0a; color:#e3b341; padding:3px 10px; border-radius:20px; font-weight:700; font-size:13px; }
    .status-4xx { background:#3d1a1a; color:#f85149; padding:3px 10px; border-radius:20px; font-weight:700; font-size:13px; }
    .status-5xx { background:#3d1a1a; color:#f85149; padding:3px 10px; border-radius:20px; font-weight:700; font-size:13px; }

    /* Method badges */
    .method-GET    { background:#0d419d; color:#58a6ff; padding:3px 10px; border-radius:6px; font-weight:700; font-size:12px; }
    .method-POST   { background:#1a4731; color:#3fb950; padding:3px 10px; border-radius:6px; font-weight:700; font-size:12px; }
    .method-PUT    { background:#3d2d0a; color:#e3b341; padding:3px 10px; border-radius:6px; font-weight:700; font-size:12px; }
    .method-PATCH  { background:#2d1a4a; color:#bc8cff; padding:3px 10px; border-radius:6px; font-weight:700; font-size:12px; }
    .method-DELETE { background:#3d1a1a; color:#f85149; padding:3px 10px; border-radius:6px; font-weight:700; font-size:12px; }

    /* URL display */
    .url-text { color:#8b949e; font-size:13px; font-family:monospace; word-break:break-all; }

    /* Timing chip */
    .timing { color:#8b949e; font-size:12px; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background:#161b22; border-radius:8px; gap:4px; }
    .stTabs [data-baseweb="tab"] { border-radius:6px; color:#8b949e; }
    .stTabs [aria-selected="true"] { background:#21262d; color:#f0f6fc; }

    /* Inputs */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background:#21262d !important;
        border: 1px solid #30363d !important;
        color:#f0f6fc !important;
        border-radius:6px !important;
    }

    /* Buttons */
    .stButton button {
        border-radius: 6px;
        font-weight: 600;
    }

    /* Section headers */
    .section-title { color:#8b949e; font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; }

    /* JSON viewer */
    .json-viewer {
        background:#0d1117;
        border: 1px solid #30363d;
        border-radius:8px;
        padding:1rem;
        font-family:monospace;
        font-size:13px;
        max-height:500px;
        overflow-y:auto;
    }

    /* Hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ───────────────────────────────────────────────────────────
def status_badge(code):
    if code is None: return ""
    if 200 <= code < 300: cls = "status-2xx"
    elif 300 <= code < 400: cls = "status-3xx"
    elif 400 <= code < 500: cls = "status-4xx"
    else: cls = "status-5xx"
    status_map = {200:"OK",201:"Created",204:"No Content",301:"Moved",302:"Found",
                  400:"Bad Request",401:"Unauthorized",403:"Forbidden",404:"Not Found",
                  405:"Method Not Allowed",429:"Too Many Requests",500:"Server Error",502:"Bad Gateway",504:"Timeout"}
    label = status_map.get(code, "")
    return f'<span class="{cls}">{code} {label}</span>'

def method_badge(method):
    cls = f"method-{method.upper()}"
    return f'<span class="{cls}">{method.upper()}</span>'

def check_backend():
    try:
        r = requests.get(f"{BACKEND}/health", timeout=3)
        return r.status_code == 200
    except:
        return False

def parse_headers_text(text):
    headers = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip()] = v.strip()
    return headers

def load_history():
    try:
        r = requests.get(f"{BACKEND}/history", timeout=5)
        return r.json() if r.status_code == 200 else []
    except:
        return []

def load_favorites():
    try:
        r = requests.get(f"{BACKEND}/favorites", timeout=5)
        return r.json() if r.status_code == 200 else []
    except:
        return []


# ── Session State ──────────────────────────────────────────────────────────────
if "response_data" not in st.session_state:
    st.session_state.response_data = None
if "prefill" not in st.session_state:
    st.session_state.prefill = {}


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ APIForge")
    st.markdown("---")

    backend_ok = check_backend()
    if backend_ok:
        st.success("🟢 Backend Connected", icon=None)
    else:
        st.error("🔴 Backend Offline — run `uvicorn backend.main:app --reload`")

    st.markdown("---")

    # Quick-load presets
    st.markdown('<div class="section-title">🧪 Quick Test APIs</div>', unsafe_allow_html=True)
    presets = {
        "PokeAPI — Pikachu": ("GET", "https://pokeapi.co/api/v2/pokemon/pikachu", {}, None),
        "JSONPlaceholder — Posts": ("GET", "https://jsonplaceholder.typicode.com/posts", {}, None),
        "Open-Meteo — Hyderabad Weather": ("GET", "https://api.open-meteo.com/v1/forecast?latitude=17.38&longitude=78.47&current_weather=true", {}, None),
        "CoinGecko — Bitcoin Price": ("GET", "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd,inr", {}, None),
        "JSONPlaceholder — Create Post": ("POST", "https://jsonplaceholder.typicode.com/posts", {}, {"title": "My Post", "body": "Hello World", "userId": 1}),
        "IP Info": ("GET", "https://ipapi.co/json/", {}, None),
    }
    for label, (m, u, h, b) in presets.items():
        if st.button(f"  {label}", key=f"preset_{label}", use_container_width=True):
            st.session_state.prefill = {"method": m, "url": u, "headers": h, "body": b}
            st.rerun()

    st.markdown("---")

    # Favorites
    st.markdown('<div class="section-title">⭐ Saved Favorites</div>', unsafe_allow_html=True)
    favorites = load_favorites()
    if favorites:
        for fav in favorites:
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(f"{fav['label']}", key=f"fav_{fav['id']}", use_container_width=True):
                    st.session_state.prefill = {
                        "method": fav["method"],
                        "url": fav["url"],
                        "headers": fav["headers"] or {},
                        "body": fav["body"],
                    }
                    st.rerun()
            with col2:
                if st.button("✕", key=f"delfav_{fav['id']}"):
                    requests.delete(f"{BACKEND}/favorites/{fav['id']}")
                    st.rerun()
    else:
        st.caption("No favorites yet. Save a request!")


# ── Main Area ──────────────────────────────────────────────────────────────────
st.markdown("# ⚡ APIForge")
st.markdown("*Your personal API testing & exploration tool*")
st.markdown("---")

tabs = st.tabs(["🚀 Request Builder", "📜 History", "📖 Learn"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — REQUEST BUILDER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    pf = st.session_state.get("prefill", {})

    col_method, col_url, col_send = st.columns([1.5, 6, 1.5])

    with col_method:
        method_idx = ["GET","POST","PUT","PATCH","DELETE"].index(pf.get("method","GET")) if pf.get("method") else 0
        method = st.selectbox("Method", ["GET","POST","PUT","PATCH","DELETE"], index=method_idx, label_visibility="collapsed")

    with col_url:
        url = st.text_input("URL", value=pf.get("url", ""), placeholder="https://api.example.com/endpoint", label_visibility="collapsed")

    with col_send:
        send = st.button("⚡ Send", type="primary", use_container_width=True)

    # ── Config Tabs ──
    cfg1, cfg2, cfg3, cfg4 = st.tabs(["🔑 Auth / Headers", "📦 Body", "🏷️ Label", "📋 Raw Request"])

    with cfg1:
        st.markdown('<div class="section-title">Auth Token (Bearer)</div>', unsafe_allow_html=True)
        token = st.text_input("Bearer Token", placeholder="Paste your token here (without 'Bearer' prefix)", type="password")

        st.markdown('<div class="section-title">Custom Headers (one per line: Key: Value)</div>', unsafe_allow_html=True)
        default_headers = pf.get("headers", {})
        default_header_text = "\n".join([f"{k}: {v}" for k, v in default_headers.items()]) if default_headers else ""
        headers_text = st.text_area("Headers", value=default_header_text, height=100, placeholder="Content-Type: application/json\nX-Custom-Header: value", label_visibility="collapsed")

    with cfg2:
        default_body = pf.get("body")
        default_body_text = json.dumps(default_body, indent=2) if default_body else ""
        body_text = st.text_area("Request Body (JSON)", value=default_body_text, height=150, placeholder='{\n  "key": "value"\n}', label_visibility="collapsed")

    with cfg3:
        request_label = st.text_input("Label (optional)", placeholder="e.g. Get all users — prod")
        save_fav = st.checkbox("⭐ Also save as favorite")

    with cfg4:
        st.caption("Preview of your request:")
        headers_parsed = parse_headers_text(headers_text)
        if token:
            headers_parsed["Authorization"] = f"Bearer {token}"
        preview = {
            "method": method,
            "url": url,
            "headers": headers_parsed,
            "body": json.loads(body_text) if body_text.strip() else None,
        }
        st.json(preview)

    # ── Send Request ──
    if send:
        if not url.strip():
            st.warning("⚠️ Please enter a URL first.")
        elif not backend_ok:
            st.error("Backend is not running. Start it first!")
        else:
            headers_parsed = parse_headers_text(headers_text)
            if token:
                headers_parsed["Authorization"] = f"Bearer {token}"

            body_parsed = None
            if body_text.strip():
                try:
                    body_parsed = json.loads(body_text)
                except:
                    st.error("❌ Invalid JSON in request body. Fix it and retry.")
                    st.stop()

            with st.spinner("Sending request..."):
                try:
                    resp = requests.post(f"{BACKEND}/forge", json={
                        "method": method,
                        "url": url,
                        "headers": headers_parsed,
                        "body": body_parsed,
                        "label": request_label,
                    }, timeout=20)
                    if resp.status_code == 200:
                        st.session_state.response_data = resp.json()
                        # Save favorite if checked
                        if save_fav and request_label:
                            requests.post(f"{BACKEND}/favorites", json={
                                "label": request_label,
                                "method": method,
                                "url": url,
                                "headers": headers_parsed,
                                "body": body_parsed,
                            })
                        st.session_state.prefill = {}
                    else:
                        detail = resp.json().get("detail","Unknown error")
                        st.error(f"❌ {detail}")
                except Exception as e:
                    st.error(f"❌ {e}")

    # ── Response Panel ──
    if st.session_state.response_data:
        rd = st.session_state.response_data
        st.markdown("---")
        st.markdown("### 📨 Response")

        # Status row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"**Status:** {status_badge(rd.get('status_code'))}", unsafe_allow_html=True)
        with m2:
            st.markdown(f"**Time:** `{rd.get('elapsed_ms')} ms`")
        with m3:
            body = rd.get("response_body", {})
            size = len(json.dumps(body))
            st.markdown(f"**Size:** `{size} bytes`")

        r1, r2 = st.tabs(["📄 Body", "📋 Headers"])

        with r1:
            st.json(rd.get("response_body", {}))

        with r2:
            resp_headers = rd.get("response_headers", {})
            if resp_headers:
                for k, v in resp_headers.items():
                    st.markdown(f"`{k}`: **{v}**")
            else:
                st.caption("No headers returned.")

        if st.button("🗑️ Clear Response"):
            st.session_state.response_data = None
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    history = load_history()

    hcol1, hcol2 = st.columns([5, 1])
    with hcol1:
        st.markdown(f"### 📜 Request History  `{len(history)} entries`")
    with hcol2:
        if st.button("🗑️ Clear All", type="secondary"):
            requests.delete(f"{BACKEND}/history")
            st.rerun()

    if not history:
        st.info("No history yet. Send your first request!")
    else:
        for item in history:
            with st.container():
                st.markdown('<div class="forge-card">', unsafe_allow_html=True)
                h1, h2, h3, h4 = st.columns([1, 5, 1.5, 1])
                with h1:
                    st.markdown(method_badge(item["method"]), unsafe_allow_html=True)
                with h2:
                    label = f"**{item['label']}** — " if item.get("label") else ""
                    st.markdown(f'{label}<span class="url-text">{item["url"]}</span>', unsafe_allow_html=True)
                with h3:
                    ts = item["timestamp"][:16].replace("T", " ")
                    st.markdown(f'<span class="timing">🕐 {ts} &nbsp; ⚡ {item["elapsed_ms"]}ms</span>', unsafe_allow_html=True)
                with h4:
                    st.markdown(status_badge(item.get("status_code")), unsafe_allow_html=True)

                with st.expander("View details / Replay"):
                    d1, d2 = st.columns(2)
                    with d1:
                        st.markdown("**Request Headers**")
                        st.json(item.get("headers", {}))
                        if item.get("body"):
                            st.markdown("**Request Body**")
                            st.json(item["body"])
                    with d2:
                        st.markdown("**Response**")
                        st.json(item.get("response_body", {}))

                    if st.button("🔄 Replay this request", key=f"replay_{item['id']}"):
                        st.session_state.prefill = {
                            "method": item["method"],
                            "url": item["url"],
                            "headers": item.get("headers", {}),
                            "body": item.get("body"),
                        }
                        st.rerun()

                    if st.button("✕ Delete", key=f"del_{item['id']}"):
                        requests.delete(f"{BACKEND}/history/{item['id']}")
                        st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LEARN
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### 📖 API Concepts Quick Reference")
    st.markdown("*Everything from the video — in one place*")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**HTTP Methods**
| Method | Purpose | Idempotent? |
|--------|---------|-------------|
| `GET` | Read data | ✅ Yes |
| `POST` | Create data | ❌ No |
| `PUT` | Replace/Update | ✅ Yes |
| `PATCH` | Partial update | ✅ Yes |
| `DELETE` | Remove data | ✅ Yes |
""")

    with c2:
        st.markdown("""
**Status Codes**
| Code | Meaning |
|------|---------|
| `200` | OK |
| `201` | Created |
| `204` | No Content |
| `400` | Bad Request |
| `401` | Unauthorized |
| `403` | Forbidden |
| `404` | Not Found |
| `500` | Server Error |
""")

    st.markdown("""
**Where does data go?**
- **Path** → identifying a specific resource: `/comments/123`
- **Query Params** → filtering/sorting: `/comments?post_id=5&sort=asc`
- **Body** → sensitive/complex data: `{ "password": "..." }`
- **Headers** → metadata & auth: `Authorization: Bearer <token>`
""")

    st.markdown("""
**API Types Cheat Sheet**
- **REST** — HTTP + JSON. Most common. Learn this first ✅
- **GraphQL** — Query language, one endpoint, flexible
- **gRPC** — Fast, uses protobuffs, great for microservices
- **SOAP** — Old, uses XML. Legacy systems only
- **WebSocket** — Bidirectional, real-time apps (chat, notifications)
""")
