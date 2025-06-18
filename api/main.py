from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import base64
import io

app = FastAPI()

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
