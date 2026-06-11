import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Import ADK requirements
from google.adk.runners import InMemoryRunner
from google.genai import types

# Assuming your agent logic is in this module
from agent_logic import root_agent 

app = FastAPI()

# Allow your React app to talk to this server
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"]
)

@app.post("/chat")
async def chat_with_agent(user_input: dict):
    query = user_input.get("message")
    
    if not query:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Initialize the runner
    runner = InMemoryRunner(agent=root_agent)
    
    # Package the string into the structured format required by the ADK
    content = types.Content(
        role='user',
        parts=[types.Part(text=query)]
    )
    
    full_response = ""
    
    try:
        # Pylance is now happy because both session_id and user_id are defined
        async for event in runner.run_async(
            session_id="web_session", 
            user_id="web_user", 
            new_message=content
        ):
            # Extract text chunks from the agent's stream
            if hasattr(event, 'content') and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        full_response += part.text
                        
        return {"reply": full_response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="Document not found or not yet generated.")
        
    return FileResponse(path=filename, filename=filename)