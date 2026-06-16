import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from google.adk.runners import InMemoryRunner
from google.genai import types

from agent_logic import (
    root_agent,
    switch_to_gemini,
    switch_to_cerebras,
    switch_to_groq,
    make_groq_model,
)

# ── Runner (persists for the life of the process) ────────────────────────────

runner: Optional[InMemoryRunner] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global runner
    runner = InMemoryRunner(agent=root_agent, app_name="react_agent_app")
    await runner.__aenter__()

    try:
        await runner.session_service.create_session(
            app_name="react_agent_app",
            user_id="web_user",
            session_id="web_session"
        )
    except Exception:
        pass

    print("[Startup] Runner initialized with Groq as primary model.", flush=True)
    yield
    await runner.__aexit__(None, None, None)

app = FastAPI(title="AI Agent Orchestrator Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Helpers ───────────────────────────────────────────────────────────────────

RATE_LIMIT_TERMS = ["429", "rate_limit", "quota", "exceeded", "resource_exhausted", "too many"]

def is_rate_limit_error(msg: str) -> bool:
    return any(term in msg.lower() for term in RATE_LIMIT_TERMS)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root_health_check():
    return {
        "status": "healthy",
        "framework": "Google ADK Engine",
        "message": "Multi-agent system orchestrator is live."
    }

@app.post("/chat")
async def chat_with_agent(user_input: dict):
    assert runner is not None, "Runner has not been initialized"

    query = user_input.get("message")
    timezone = user_input.get("timezone", "UTC")
    local_time = user_input.get("local_time", "unknown")

    if not query:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    enriched_query = f"""[User context: Current timezone is {timezone}, local time is {local_time}] User request: {query}"""

    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        if not os.environ.get("GROQ_API_KEY") and not os.environ.get("CEREBRAS_API_KEY"):
            raise HTTPException(status_code=500, detail="No API keys configured.")

    user_id = "web_user"
    session_id = "web_session"

    # Ensure session exists
    try:
        session = await runner.session_service.get_session(
            app_name="react_agent_app",
            user_id=user_id,
            session_id=session_id
        )
        if session is None:
            await runner.session_service.create_session(
                app_name="react_agent_app",
                user_id=user_id,
                session_id=session_id
            )
    except Exception:
        await runner.session_service.create_session(
            app_name="react_agent_app",
            user_id=user_id,
            session_id=session_id
        )

    content = types.Content(
        role='user',
        parts=[types.Part(text=enriched_query)]
    )

    async def run_agent() -> str:
        """Execute the agent and collect the full response."""

        assert runner is not None, "Runner has not been initialized"
        full_response = ""

        async for event in runner.run_async(
            session_id=session_id,
            user_id=user_id,
            new_message=content
        ):
            if hasattr(event, 'content') and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        full_response += part.text
        return full_response

    # ── 4-model fallback chain ────────────────────────────────────────────────
    # Order: Groq (key 1) → Groq (key 2, rotated) → Gemini → Cerebras
    fallback_chain = [
        ("Groq (primary key)",   None),             # already active at startup
        ("Groq (rotated key)",   switch_to_groq),   # rotates to key 2 if available
        ("Gemini 2.0 Flash",     switch_to_gemini),
        ("Cerebras llama-3.3",   switch_to_cerebras),
    ]

    last_error = None

    for model_name, switch_fn in fallback_chain:
        if switch_fn:
            switch_fn()
        try:
            print(f"[Model Router] Trying {model_name}...", flush=True)
            result = await run_agent()
            print(f"[Model Router] Success with {model_name}", flush=True)
            return {"reply": result}

        except Exception as e:
            last_error = e
            error_str = str(e)

            if is_rate_limit_error(error_str):
                print(f"[Model Router] {model_name} rate limited → trying next...", flush=True)
                continue

            # Non-rate-limit error — log and raise immediately, don't try other models
            print(f"RUNTIME AGENT ERROR: {error_str}", file=sys.stderr, flush=True)
            raise HTTPException(status_code=500, detail=f"Agent Execution Failure: {error_str}")

    # All 4 models exhausted
    print("[Model Router] All models rate limited.", file=sys.stderr, flush=True)
    raise HTTPException(
        status_code=429,
        detail="All models are currently rate limited. Please wait a few minutes and try again."
    )

@app.get("/download/{filename}")
async def download_file(filename: str):
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="The requested document has not been compiled yet.")
    return FileResponse(
        path=filename,
        filename=filename,
        media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    )