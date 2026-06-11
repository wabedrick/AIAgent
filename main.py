
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Import Google ADK requirements
from google.adk.runners import InMemoryRunner
from google.genai import types

# Import your underlying agent orchestration logic
from agent_logic import root_agent 

app = FastAPI(title="AI Agent Orchestrator Backend")

# Robust CORS middleware configuration to allow your React application to connect
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"]
)

# -------------------------------------------------------------
# 🔥 ADDED: Root health check route to eliminate Render 404 logs
# -------------------------------------------------------------
@app.get("/")
async def root_health_check():
    """
    Catches background health pings and browser visits to verify 
    the web container is operational.
    """
    return {
        "status": "healthy",
        "framework": "Google ADK Engine",
        "message": "Multi-agent system orchestrator is live."
    }

# @app.post("/chat")
# async def chat_with_agent(user_input: dict):
#     # Safely parse the JSON payload sent by your React component
#     query = user_input.get("message")
    
#     if not query:
#         raise HTTPException(status_code=400, detail="Incoming request message payload cannot be empty.")

#     # Diagnostic check for critical environment variables before execution
#     if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
#         print("CRITICAL CONFIG ERROR: Missing Google Gemini API Key in environment settings.", file=sys.stderr, flush=True)
#         raise HTTPException(status_code=500, detail="Backend configuration error: Gemini API key is missing on Render settings.")

#     try:
#         # Initialize the stateful memory runner for the agent
#         runner = InMemoryRunner(agent=root_agent)
        
#         # Package the raw input string into the structured schemas required by the Google GenAI client
#         content = types.Content(
#             role='user',
#             parts=[types.Part(text=query)]
#         )
        
#         full_response = ""
        
#         # Stream the multi-turn execution events asynchronously
#         async for event in runner.run_async(
#             session_id="web_session", 
#             user_id="web_user", 
#             new_message=content
#         ):
#             # Inspect and safely concatenate the chunked streaming text fragments
#             if hasattr(event, 'content') and event.content and event.content.parts:
#                 for part in event.content.parts:
#                     if hasattr(part, 'text') and part.text:
#                         full_response += part.text
                        
#         return {"reply": full_response}
        
#     except Exception as e:
#         # Force print the exact Python error traceback straight to Render logs for visibility
#         print(f"RUNTIME AGENT ERROR EXCEPTION: {str(e)}", file=sys.stderr, flush=True)
#         # Propagate the raw internal error message text back to your React client for easy debugging
#         raise HTTPException(status_code=500, detail=f"Agent Execution Failure: {str(e)}")
@app.post("/chat")
async def chat_with_agent(user_input: dict):
    query = user_input.get("message")
    
    if not query:
        raise HTTPException(status_code=400, detail="Incoming request message payload cannot be empty.")

    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("CRITICAL CONFIG ERROR: Missing Google API Key in environment settings.", file=sys.stderr, flush=True)
        raise HTTPException(status_code=500, detail="Backend configuration error: API key is missing.")

    try:
        # 🔥 THE REAL FIX: Use 'async with' to properly boot the runner's internal session services.
        # Without this context manager, the internal memory DB remains closed and throws 500 errors.
        async with InMemoryRunner(agent=root_agent) as runner:
            
            session_id = "web_session"
            user_id = "web_user"
            
            # Safely attempt to allocate the session now that the internal DB is booted
            try:
                await runner.session_service.create_session(
                    app_name="react_agent_app",
                    user_id=user_id,
                    session_id=session_id
                )
            except Exception:
                # If the ADK auto-created the session behind the scenes, we safely ignore the creation error
                pass 
            
            # Package the input into the strict ADK schema
            content = types.Content(
                role='user',
                parts=[types.Part(text=query)]
            )
            
            full_response = ""
            
            # Execute the stream with the strict Pylance parameters
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
        print(f"RUNTIME AGENT ERROR EXCEPTION: {str(e)}", file=sys.stderr, flush=True)
        raise HTTPException(status_code=500, detail=f"Agent Execution Failure: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    # Verify the requested asset generated by your agent logic exists on disk
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="The requested document has not been compiled yet.")
        
    return FileResponse(path=filename, filename=filename, media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document')



# import os
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse

# # Import ADK requirements
# from google.adk.runners import InMemoryRunner
# from google.genai import types

# # Assuming your agent logic is in this module
# from agent_logic import root_agent 

# app = FastAPI()

# # Allow your React app to talk to this server
# app.add_middleware(
#     CORSMiddleware, 
#     allow_origins=["*"], 
#     allow_credentials=True,
#     allow_methods=["*"], 
#     allow_headers=["*"]
# )

# @app.post("/chat")
# async def chat_with_agent(user_input: dict):
#     query = user_input.get("message")
    
#     if not query:
#         raise HTTPException(status_code=400, detail="Message cannot be empty.")

#     # Initialize the runner
#     runner = InMemoryRunner(agent=root_agent)
    
#     # Package the string into the structured format required by the ADK
#     content = types.Content(
#         role='user',
#         parts=[types.Part(text=query)]
#     )
    
#     full_response = ""
    
#     try:
#         # Pylance is now happy because both session_id and user_id are defined
#         async for event in runner.run_async(
#             session_id="web_session", 
#             user_id="web_user", 
#             new_message=content
#         ):
#             # Extract text chunks from the agent's stream
#             if hasattr(event, 'content') and event.content and event.content.parts:
#                 for part in event.content.parts:
#                     if hasattr(part, 'text') and part.text:
#                         full_response += part.text
                        
#         return {"reply": full_response}
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/download/{filename}")
# async def download_file(filename: str):
#     if not os.path.exists(filename):
#         raise HTTPException(status_code=404, detail="Document not found or not yet generated.")
        
#     return FileResponse(path=filename, filename=filename)