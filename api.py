import asyncio
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from agents import run_pipeline

import os

app = FastAPI(title="Role Analysis API", version="1.0.0")

origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Accept a .docx resume upload and stream progress + final results back
    as newline-delimited JSON (NDJSON).

    Each streamed line is a JSON object:
      { "type": "progress", "step": int, "message": str }
      { "type": "result",   "extract": str, "analysis": str, "advice": str }
      { "type": "error",    "message": str }
    """
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported.")

    file_bytes = await file.read()

    # We use a queue to bridge the sync pipeline thread and the async generator
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def progress_callback(step: int, message: str):
        event = json.dumps({"type": "progress", "step": step, "message": message})
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def run_in_thread():
        try:
            result = await asyncio.to_thread(run_pipeline, file_bytes, progress_callback)
            event = json.dumps({"type": "result", **result})
            await queue.put(event)
        except Exception as e:
            event = json.dumps({"type": "error", "message": str(e)})
            await queue.put(event)
        finally:
            await queue.put(None)  # sentinel

    async def event_stream():
        task = asyncio.create_task(run_in_thread())
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item + "\n"
        await task

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
