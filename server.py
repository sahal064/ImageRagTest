import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.runner import run_agent

app = FastAPI(title="Image Search + Reasoning API")

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


class SearchRequest(BaseModel):
    query: str


@app.post("/api/search")
async def search(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = await asyncio.to_thread(run_agent, request.query)

    for img in result.get("images", []):
        raw = img.get("path", "")
        img["url"] = "/" + raw.replace("\\", "/")

    return result


@app.get("/api/images")
async def list_images():
    files = sorted(
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    return {"images": [{"name": f, "url": f"/images/{f}"} for f in files]}


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
