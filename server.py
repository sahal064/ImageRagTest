import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

import asyncio
from fastapi import FastAPI, File, HTTPException, UploadFile, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import secrets

from agent.runner import run_agent
from retrieval.search import search_by_image

app = FastAPI(title="Image Search + Reasoning API")

# Authentication credentials (static for now)
VALID_USERNAME = "sahal"
VALID_PASSWORD = "1234"
SESSION_TOKEN = secrets.token_hex(32)

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


def is_authenticated(request: Request) -> bool:
    """Check if user has valid session token"""
    token = request.cookies.get("session_token")
    return token == SESSION_TOKEN


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/login")
async def login(creds: LoginRequest, response: Response):
    """Login endpoint"""
    if creds.username == VALID_USERNAME and creds.password == VALID_PASSWORD:
        # Set session cookie (httponly=True for security, max_age=7 days)
        response.set_cookie(
            "session_token",
            SESSION_TOKEN,
            max_age=7 * 24 * 60 * 60,
            httponly=True,
            samesite="lax"
        )
        return {"status": "success"}
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Image Search + Reasoning API"}


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


def _clamp_top_k(value: int) -> int:
    return max(1, min(int(value), 50))


@app.post("/api/search")
async def search(request: SearchRequest, http_request: Request):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    top_k = _clamp_top_k(request.top_k)
    result = await asyncio.to_thread(run_agent, request.query, top_k)

    for img in result.get("images", []):
        raw = img.get("path", "")
        img["url"] = "/" + raw.replace("\\", "/")

    return result


@app.post("/api/search-by-image")
async def search_by_image_endpoint(request: Request, file: UploadFile = File(...), top_k: int = 5):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()
    results = await asyncio.to_thread(search_by_image, image_bytes, _clamp_top_k(top_k))

    for img in results:
        raw = img.get("path", "")
        img["url"] = "/" + raw.replace("\\", "/")

    return {"images": results}


@app.get("/api/images")
async def list_images(request: Request):
    files = sorted(
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    return {"images": [{"name": f, "url": f"/images/{f}"} for f in files]}


@app.get("/")
async def root(request: Request):
    if not is_authenticated(request):
        return FileResponse(os.path.join(STATIC_DIR, "login.html"))
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
