import base64, io, json, os, time
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image
import pytesseract        # pip install pytesseract pillow
import openai             # pip install openai>=1.0.0
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from embeddings import load_embeddings, get_openai_embedding, client as _shared_client # type: ignore[import]

from fastapi.middleware.cors import CORSMiddleware



# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _image_to_text(b64_img: Optional[str]) -> str:
    """OCR any base‑64‑encoded image; return extracted text or empty string."""
    if not b64_img:
        return ""
    try:
        img_bytes = base64.b64decode(b64_img)
        img = Image.open(io.BytesIO(img_bytes))
        return pytesseract.image_to_string(img)
    except Exception:
        # If decoding fails just ignore image; continue with text only
        return ""


def _get_embedding(text: str, *, model: str = None) -> np.ndarray:
    """Proxy to the user‑supplied *get_openai_embedding* helper and L2‑normalise."""
    vec_list = get_openai_embedding(text)  # type: ignore[arg-type]
    if vec_list is None:
        raise RuntimeError("Failed to obtain embedding for the input text.")
    vec = np.asarray(vec_list, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def _top_k_indices(query_vec: np.ndarray, corpus_vecs: np.ndarray, k: int = 10) -> np.ndarray:
    """Fast cosine‑similarity search using a cached, normalised corpus matrix."""
    if not hasattr(_top_k_indices, "_norm_corpus"):
        _top_k_indices._norm_corpus = (
            corpus_vecs / np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
        )
    sims = _top_k_indices._norm_corpus @ query_vec
    return sims.argsort()[-k:][::-1]

# ---------------------------------------------------------------------------
# Core QA logic
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    image_b64: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    k: int = 10,
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    """Return concise GPT‑4o answer plus *k* supporting passage indices."""
    # Prepare an OpenAI client compatible with >=1.0.0 SDK
    if _shared_client is not None:
        client = _shared_client  # reuse the one from the embedding helper
    else:
        client = openai.OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))

    # Combine question + OCR text
    combined_text = question.strip()
    ocr_text = _image_to_text(image_b64)
    if ocr_text:
        combined_text += "\n\nImage text:\n" + ocr_text.strip()

    # Embed query
    q_vec = _get_embedding(combined_text)

    # Retrieve passages
    vectors, metadata = load_embeddings()
    top_idxs = _top_k_indices(q_vec, vectors, k=k)

    # Craft GPT prompt
    context_blocks = [f"[{i}] {metadata[i]['text_chunk']}" for i in top_idxs]
    prompt = (
        "You are the TDS Virtual TA. Answer the student's question concisely, "
        "citing the passage indices in brackets when appropriate.\n\n"
        f"Question:\n{question}\n\n"
        "Relevant passages:\n" + "\n".join(context_blocks)
    )
    chat_resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer_text = chat_resp.choices[0].message.content.strip()

    # Build links
    links: List[Dict[str, Any]] = [
        {
            "index": int(i),
            "text": metadata[i].get("title", ""),
            "url": metadata[i].get("url", ""),
        }
        for i in top_idxs
    ]

    return {"answer": answer_text, "links": links}

# ---------------------------------------------------------------------------
# FastAPI setup
# ---------------------------------------------------------------------------

app = FastAPI(title="TDS Virtual TA API", docs_url="/docs", redoc_url="/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify domains
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],
)
class QARequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64‑encoded image string

class QAResponse(BaseModel):
    answer: str
    links: List[Dict[str, Any]]

@app.post("/api/ta", response_model=QAResponse)
async def qa_endpoint(payload: QARequest):
    start = time.time()
    
    try:
        result = answer_question(payload.question, image_b64=payload.image)
        duration = time.time() - start
        # Ensure we adhere to the 30‑second budget (log only)
        if duration > 30:
            print(f"Warning: response took {duration:.1f}s")
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# ---------------------------------------------------------------------------
# Run with: `python thisfile.py` ---------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
