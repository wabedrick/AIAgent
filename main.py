import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from google.adk.runners import InMemoryRunner
from google.genai import types

from agent_logic import root_agent, switch_to_gemini, switch_to_groq

# --- Module-level runner (persists for the life of the process) ---
runner: InMemoryRunner | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Boot the runner once on startup, tear it down on shutdown."""
    global runner
    runner = InMemoryRunner(agent=root_agent, app_name="react_agent_app")
    await runner.__aenter__()

    # Pre-create the session so it's ready for the first request
    try:
        await runner.session_service.create_session(
            app_name="react_agent_app",
            user_id="web_user",
            session_id="web_session"
        )
    except Exception:
        pass  # Already exists

    yield  # App is now running

    # Cleanup on shutdown
    await runner.__aexit__(None, None, None)

app = FastAPI(title="AI Agent Orchestrator Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

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
    if not query:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    user_id = "web_user"
    session_id = "web_session"

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
        parts=[types.Part(text=query)]
    )

    full_response = ""

    try:
        async for event in runner.run_async(
            session_id=session_id,
            user_id=user_id,
            new_message=content
        ):
            if hasattr(event, 'content') and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        full_response += part.text

        return {"reply": full_response}

    except Exception as e:
        error_msg = str(e).lower()

        # Groq rate limit hit → switch to Gemini and retry once
        if any(term in error_msg for term in ["429", "rate_limit", "quota", "exceeded"]):
            print(f"[Model Router] Rate limit hit, switching models...", flush=True)
            switch_to_gemini()

            try:
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

                return {"reply": full_response}

            except Exception as e2:
                error_msg2 = str(e2).lower()
                # Gemini also rate limited → tell user to wait
                if any(term in error_msg2 for term in ["429", "rate_limit", "quota", "exceeded"]):
                    raise HTTPException(
                        status_code=429,
                        detail="Both models are rate limited. Please wait a few minutes and try again."
                    )
                raise HTTPException(status_code=500, detail=f"Agent Execution Failure: {str(e2)}")

        print(f"RUNTIME AGENT ERROR EXCEPTION: {str(e)}", file=sys.stderr, flush=True)
        raise HTTPException(status_code=500, detail=f"Agent Execution Failure: {str(e)}")

# async def chat_with_agent(user_input: dict):
#     assert runner is not None, "Runner has not been initialized"
#     query = user_input.get("message")

#     if not query:
#         raise HTTPException(status_code=400, detail="Incoming request message payload cannot be empty.")

#     if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
#         print("CRITICAL CONFIG ERROR: Missing Google API Key.", file=sys.stderr, flush=True)
#         raise HTTPException(status_code=500, detail="Backend configuration error: API key is missing.")

#     user_id = "web_user"
#     session_id = "web_session"

#     # Ensure session exists (handles cases where runner restarted mid-flight)
#     try:
#         session = await runner.session_service.get_session(
#             app_name="react_agent_app",
#             user_id=user_id,
#             session_id=session_id
#         )
#         if session is None:
#             await runner.session_service.create_session(
#                 app_name="react_agent_app",
#                 user_id=user_id,
#                 session_id=session_id
#             )
#     except Exception:
#         await runner.session_service.create_session(
#             app_name="react_agent_app",
#             user_id=user_id,
#             session_id=session_id
#         )

#     content = types.Content(
#         role='user',
#         parts=[types.Part(text=query)]
#     )

#     full_response = ""

#     try:
#         async for event in runner.run_async(
#             session_id=session_id,
#             user_id=user_id,
#             new_message=content
#         ):
#             if hasattr(event, 'content') and event.content and event.content.parts:
#                 for part in event.content.parts:
#                     if hasattr(part, 'text') and part.text:
#                         full_response += part.text

#         return {"reply": full_response}

#     except Exception as e:
#         print(f"RUNTIME AGENT ERROR EXCEPTION: {str(e)}", file=sys.stderr, flush=True)
#         raise HTTPException(status_code=500, detail=f"Agent Execution Failure: {str(e)}")


@app.get("/download/{filename}")
async def download_file(filename: str):
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="The requested document has not been compiled yet.")
    return FileResponse(
        path=filename,
        filename=filename,
        media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    )