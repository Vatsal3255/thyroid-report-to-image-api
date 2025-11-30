# backend/app/main.py
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from io import BytesIO

from app.config_store import save_api_key, load_api_key

from app import auth as auth_router
from app import reports as reports_router
from app.json_extract import extract_info, YOUR_API_KEY
from app.generate_image import generate_image as render_thyroid_image

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Image Gen Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(reports_router.router)

class EchoIn(BaseModel):
    message: str

class APIKeyIn(BaseModel):
    key: str

class ExtractJSONIn(BaseModel):
    report_text: str


class GenerateImageIn(BaseModel):
    report_text: str
    extracted_json: Dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/echo")
def echo(payload: EchoIn):
    return {"you_sent": payload.message}

# ---- KEY MANAGEMENT ENDPOINTS ----
@app.post("/save-key")
def save_key(payload: APIKeyIn):
    k = payload.key.strip()
    if not k:
        raise HTTPException(status_code=400, detail="Key cannot be empty")
    save_api_key(k)
    return {"message": "API key saved"}

@app.get("/get-key")
def get_key():
    k = load_api_key()
    if not k:
        raise HTTPException(status_code=404, detail="No API key found")
    # return the raw key; or mask it if you prefer
    return {"GENAI_API_KEY": k}

@app.post("/extract-json")
async def extract_json(payload: ExtractJSONIn):
    text = (payload.report_text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="report_text is required")
    api_key = YOUR_API_KEY or load_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="GenAI API key not configured")
    try:
        result = await run_in_threadpool(extract_info, text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction error: {exc}") from exc
    if not result:
        raise HTTPException(status_code=500, detail="Unable to extract info")
    return {"extracted": result}

# ---- IMAGE GENERATION ----

@app.post("/generate-image")
async def generate_image(payload: GenerateImageIn):
    if not payload.extracted_json:
        raise HTTPException(status_code=400, detail="extracted_json is required")

    try:
        image = render_thyroid_image(payload.extracted_json)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to render image: {exc}") from exc

    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
