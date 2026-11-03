import os, time, collections
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


@app.post("/api/chat")
async def chat(body: Chat, req: Request):
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
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    try:
        async with httpx.AsyncClient(timeout=40) as c:
            r = await c.post(url, json=payload, headers={"x-goog-api-key": API_KEY})
    except httpx.HTTPError:
        raise HTTPException(502, "Gemini'ye ulaşılamadı.")
    if r.status_code != 200:
        raise HTTPException(502, f"Gemini hatası ({r.status_code}).")
    try:
        parts = r.json()["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError):
        text = ""
    return {"reply": text or "Buna bir cevap üretemedim, tekrar dener misin?"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
