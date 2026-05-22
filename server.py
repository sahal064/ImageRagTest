import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

import asyncio  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402
import secrets  # noqa: E402

from agent.runner import run_agent  # noqa: E402
from retrieval.search import search_by_image  # noqa: E402

app = FastAPI(title="Image Search + Reasoning API")

VALID_USERNAME = "sahal"
VALID_PASSWORD = "1234"
active_tokens: dict[str, str] = {}  # token -> username

bearer_scheme = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGES_DIR = os.path.join(PROJECT_ROOT, os.getenv("IMAGES_DIR", "images"))
STATIC_DIR = os.path.join(PROJECT_ROOT, "ui", "static")

app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    token = credentials.credentials
    if token not in active_tokens:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return token


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/login")
async def login(creds: LoginRequest):
    if creds.username == VALID_USERNAME and creds.password == VALID_PASSWORD:
        token = secrets.token_hex(32)
        active_tokens[token] = creds.username
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid username or password")


@app.post("/api/logout")
async def logout(token: str = Depends(require_auth)):
    active_tokens.pop(token, None)
    return {"status": "logged out"}


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


def _clamp_top_k(value: int) -> int:
    return max(1, min(int(value), 50))


@app.post("/api/search")
async def search(request: SearchRequest, _: str = Depends(require_auth)):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    top_k = _clamp_top_k(request.top_k)
    result = await asyncio.to_thread(run_agent, request.query, top_k)

    for img in result.get("images", []):
        raw = img.get("path", "")
        img["url"] = "/" + raw.replace("\\", "/")

    return result


@app.post("/api/search-by-image")
async def search_by_image_endpoint(
    file: UploadFile = File(...),
    top_k: int = 5,
    _: str = Depends(require_auth),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Uploaded file must be an image"
        )

    image_bytes = await file.read()
    top_k_clamped = _clamp_top_k(top_k)
    results = await asyncio.to_thread(search_by_image, image_bytes, top_k_clamped)

    for img in results:
        raw = img.get("path", "")
        img["url"] = "/" + raw.replace("\\", "/")

    return {"images": results}


@app.get("/api/images")
async def list_images(_: str = Depends(require_auth)):
    files = sorted(
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    return {"images": [{"name": f, "url": f"/images/{f}"} for f in files]}


@app.get("/login")
async def login_page():
    return FileResponse(os.path.join(STATIC_DIR, "login.html"))


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
