from fastapi import FastAPI, UploadFile, Request
from pydantic import BaseModel
import base64
import io
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
from pathlib import Path

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    image: str | None = None

@app.post("/api/")
async def answer_query(query: Query):
    # Decode image (optional)
    if query.image:
        image_bytes = base64.b64decode(query.image)
        # OCR logic here (e.g. pytesseract)

    # Perform semantic search + LLM prompt
    answer, links = get_answer_from_rag(query.question)
    return {
        "answer": answer,
        "links": links
    }
