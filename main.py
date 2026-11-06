
import os
import asyncio
import logging

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DAYDREAM_API_KEY = os.getenv("DAYDREAM_API_KEY")
DAYDREAM_BASE_URL = "https://gen.daydream.live/v1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moodmix")

app = FastAPI(
    title="MoodMix Backend",
    description="Backend API for MoodMix music generation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Request models
# --------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=400)
    max_new_tokens: int = 256
    generate_lyrics: bool = False
    genre: str | None = None
    bpm: int | None = Field(default=None, ge=60, le=200)


# --------------------------------------------------
# Shared Daydream API helper
# --------------------------------------------------

async def daydream_request(method: str, path: str, **kwargs):
    if not DAYDREAM_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Daydream API key is not configured on the server."
        )

    headers = {
        "Authorization": f"Bearer {DAYDREAM_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.request(
                method,
                f"{DAYDREAM_BASE_URL}{path}",
                headers=headers,
                **kwargs
            )

        if response.status_code >= 400:
            logger.error(
                "Daydream API error %s: %s",
                response.status_code,
                response.text[:1000]
            )

            if response.status_code == 401:
                raise HTTPException(502, "Daydream rejected the API key.")
            if response.status_code == 402:
                raise HTTPException(402, "Daydream account has insufficient credits.")
            if response.status_code == 429:
                raise HTTPException(429, "Daydream rate limit reached.")
            if response.status_code == 503:
                raise HTTPException(503, "Daydream service is temporarily unavailable.")

            raise HTTPException(
                502,
                f"Daydream API returned status {response.status_code}."
            )

        return response.json()

    except httpx.TimeoutException:
        raise HTTPException(504, "Daydream API request timed out.")
    except httpx.RequestError:
        logger.exception("Could not connect to Daydream")
        raise HTTPException(502, "Could not connect to Daydream.")


# --------------------------------------------------
# Health checks
# --------------------------------------------------

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "MoodMix Backend",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


# --------------------------------------------------
# Music generation
# --------------------------------------------------

@app.post("/generate")
async def generate_music(request: GenerateRequest):

    payload = {
        "prompt": request.prompt,
        "instrumental": not request.generate_lyrics,
    }

    if request.bpm is not None:
        payload["bpm"] = request.bpm

    if request.genre:
        payload["prompt"] = f"{request.genre} music. {request.prompt}"

    result = await daydream_request(
        "POST",
        "/songs",
        json=payload
    )

    song_id = result.get("id") or result.get("song_id")

    if not song_id:
        logger.error("Unexpected Daydream response: %s", result)
        raise HTTPException(502, "Daydream did not return a song ID.")

    return {
        "status": result.get("status", "queued"),
        "song_id": song_id,
        "song_title": result.get("title", "Untitled"),
        "audio_url": None,
        "lyrics": result.get("lyrics"),
        "message": "Your music generation request has been submitted."
    }


# --------------------------------------------------
# Check generation status
# --------------------------------------------------

@app.get("/status/{song_id}")
async def get_song_status(song_id: str):

    result = await daydream_request(
        "GET",
        f"/songs/{song_id}"
    )

    takes = result.get("takes") or []
    audio_url = None

    if takes:
        audio_url = takes[0].get("mp3_url") or takes[0].get("wav_url")

    return {
        "status": result.get("status", "processing"),
        "song_id": result.get("id") or song_id,
        "song_title": result.get("title"),
        "audio_url": audio_url,
        "error": result.get("error"),
        "message": result.get("message")
    }


# --------------------------------------------------
# Generate and wait for completion
# --------------------------------------------------

@app.post("/generate-and-wait")
async def generate_and_wait(request: GenerateRequest):

    payload = {
        "prompt": request.prompt,
        "instrumental": not request.generate_lyrics,
    }

    if request.bpm is not None:
        payload["bpm"] = request.bpm

    if request.genre:
        payload["prompt"] = f"{request.genre} music. {request.prompt}"

    result = await daydream_request(
        "POST",
        "/songs",
        json=payload
    )

    song_id = result.get("id") or result.get("song_id")

    if not song_id:
        raise HTTPException(502, "Daydream did not return a song ID.")

    for _ in range(60):
        await asyncio.sleep(2)

        status_result = await daydream_request(
            "GET",
            f"/songs/{song_id}"
        )

        takes = status_result.get("takes") or []
        audio_url = None

        if takes:
            audio_url = takes[0].get("mp3_url") or takes[0].get("wav_url")

        status = status_result.get("status", "processing")

        if status in ("completed", "success"):
            return {
                "status": status,
                "song_id": song_id,
                "song_title": status_result.get("title", "Untitled"),
                "audio_url": audio_url,
                "lyrics": status_result.get("lyrics"),
                "message": "Music generated successfully."
            }

        if status in ("failed", "error"):
            raise HTTPException(
                502,
                status_result.get("error", "Music generation failed.")
            )

    return {
        "status": "processing",
        "song_id": song_id,
        "message": "Generation is taking longer than expected. Check the status endpoint."
    }


# --------------------------------------------------
# Daydream usage / credits
# --------------------------------------------------

@app.get("/usage")
async def get_usage():
    return await daydream_request("GET", "/usage")
