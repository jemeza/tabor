"""
FastAPI server exposing the Tabor research agent via SSE streaming.

Run with:
    uvicorn api:app --reload --port 8000
"""

import asyncio
import json
import queue as thread_queue
import threading
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

from agent import build_agent  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    _agent = build_agent()
    yield


app = FastAPI(title="Tabor Research API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    query: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/research")
async def research(request: ResearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    q: thread_queue.Queue = thread_queue.Queue()

    def run_in_thread():
        try:
            inputs = {"messages": [HumanMessage(content=request.query)]}
            final_content = ""
            for chunk in _agent.stream(inputs, stream_mode="values"):
                last_msg = chunk["messages"][-1]
                if hasattr(last_msg, "content") and isinstance(last_msg.content, str):
                    if last_msg.content and last_msg.content != final_content:
                        new_text = last_msg.content[len(final_content):]
                        q.put({"text": new_text})
                        final_content = last_msg.content
        except Exception as e:
            q.put({"error": str(e)})
        finally:
            q.put(None)  # sentinel

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()

    async def generate():
        loop = asyncio.get_event_loop()
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                yield "data: [DONE]\n\n"
                break
            if "error" in item:
                yield f"data: {json.dumps({'error': item['error']})}\n\n"
                yield "data: [DONE]\n\n"
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
