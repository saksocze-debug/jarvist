eimport os, time, collections
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
LIMIT = int(os.environ.get("RATE_LIMIT_PER_MIN", "15"))
SYSTEM = ("Sen MARK, holografik bir yapay zeka asistanısın. Kullanıcının dilinde "
          "(varsayılan Türkçe) cevap ver. Cevapların sesli okunacak: kısa, akıcı, "
          "doğal konuş; markdown, emoji ve liste kullanma.")

app = FastAPI(title="MARK Hologram")
hits = collections.defaultdict(list)


class Msg(BaseModel):
    role: str = Field(pattern="^(user|model)$")
    text: str = Field(max_length=2000)


class Chat(BaseModel):
    messages: list[Msg] = Field(min_length=1, max_length=20)


def client_ip(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "?")


@app.get("/healthz")
def healthz():
    return {"ok": True}


BASE = "https://generativelanguage.googleapis.com/v1beta"
BAD = ("lite", "image", "tts", "live", "embed", "audio", "robotics", "computer", "vision")
_model = None


async def pick_model(c: httpx.AsyncClient):
    """Bu anahtarla gerçekten çalışan en yeni Flash modelini bulur."""
    r = await c.get(f"{BASE}/models", params={"pageSize": 200}, headers={"x-goog-api-key": API_KEY})
    if r.status_code != 200:
        return None
    names = []
    for m in r.json().get("models", []):
        n = m.get("name", "").removeprefix("models/")
        if ("generateContent" in m.get("supportedGenerationMethods", [])
                and n.startswith("gemini") and not any(b in n for b in BAD)):
            names.append(n)
    flash = sorted([n for n in names if "flash" in n], reverse=True)
    return (flash or sorted(names, reverse=True) or [None])[0]


def gerr(r: httpx.Response) -> str:
    try:
        return r.json()["error"]["message"][:200]
    except Exception:
        return r.text[:200]


@app.post("/api/chat")
async def chat(body: Chat, req: Request):
    global _model
    if not API_KEY:
        raise HTTPException(500, "Sunucuda GEMINI_API_KEY tanımlı değil.")
    if body.messages[0].role != "user" or body.messages[-1].role != "user":
        raise HTTPException(400, "Geçersiz mesaj sırası.")
    now, ip = time.time(), client_ip(req)
    hits[ip] = [t for t in hits[ip] if now - t < 60]
    if len(hits[ip]) >= LIMIT:
        raise HTTPException(429, "Çok hızlı gidiyorsun, bir dakika bekle.")
    hits[ip].append(now)

    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": m.role, "parts": [{"text": m.text}]} for m in body.messages],
        "generationConfig": {"temperature": 0.8},
    }
    hdr = {"x-goog-api-key": API_KEY}
    try:
        async with httpx.AsyncClient(timeout=40) as c:
            model = _model or MODEL
            r = await c.post(f"{BASE}/models/{model}:generateContent", json=payload, headers=hdr)
            if r.status_code == 404:
                found = await pick_model(c)
                if found and found != model:
                    _model = model = found
                    r = await c.post(f"{BASE}/models/{model}:generateContent", json=payload, headers=hdr)
    except httpx.HTTPError:
        raise HTTPException(502, "Gemini'ye ulaşılamadı.")
    if r.status_code != 200:
        raise HTTPException(502, f"Gemini {r.status_code} ({model}): {gerr(r)}")
    try:
        parts = r.json()["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError):
        text = ""
    return {"reply": text or "Buna bir cevap üretemedim, tekrar dener misin?"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
