
import os
import platform
from docx import Document
from dotenv import load_dotenv
from duckduckgo_search import DDGS

# Import native Google ADK requirements
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool, AgentTool, google_search
from google.genai import types
from google.adk.models.lite_llm import LiteLlm
import litellm


litellm.set_verbose = True  # shows full error details in Render logs

load_dotenv()

# 1. Retrieve the model identifier set in your Render environment variables panel.
# We fall back to "gemini-2.5-flash" if running locally without an env file.
# my_model_name = os.getenv("my_model_name", "gemini-2.0-flash")

# my_model = LiteLlm(model="groq/llama-3.1-8b-instant")

my_model = LiteLlm(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    num_retries=5,
    timeout=60
)

# 2. Initialize the native Google Gemini model configuration.
# The underlying Google GenAI SDK automatically grabs your API key from the environment.
# my_model = Gemini(model=my_model_name)


def web_search(query: str) -> str:
    """Use this tool to search the internet for current events, facts, or data."""
    try:
        # Initialize DuckDuckGo search
        ddgs = DDGS()
        # Get the top 3 results
        results = ddgs.text(query, max_results=3)
        
        # Format the results into a string for the LLM to read
        formatted_results = "\n\n".join(
            [f"Title: {res['title']}\nSnippet: {res['body']}" for res in results]
        )
        return formatted_results if formatted_results else "No results found."
    except Exception as e:
        return f"Search failed: {str(e)}"

# Wrap the standard function into an ADK-compatible tool
my_search_tool = FunctionTool(web_search)

# Research Agent: Its job is to use the custom web_search tool and present findings.
research_agent = Agent(
    name="ResearchAgent",
    model=my_model,
    instruction="""You are a specialized research agent. Your only job is to use the
    web_search tool to find 2-3 pieces of relevant information on the given topic and present the findings with citations.""",
    tools=[my_search_tool], 
    output_key="research_findings",  # Stored in the session state
)

# Summarizer Agent: Its job is to summarize the text it receives.
summarizer_agent = Agent(
    name="SummarizerAgent",
    model=my_model,
    instruction="""Read the provided research findings: {research_findings}
Create a concise summary as a bulleted list with 3-5 key points.""",
    output_key="final_summary",
)

# This function will be used by the CVStylistAgent to save the generated markdown as a Word document.
def save_to_word(text_content: str, filename: str = "Document.docx") -> str:
    """
    Creates a formatted Word document from markdown text.
    
    CRITICAL INSTRUCTION FOR LLM:
    - `text_content`: The FULL markdown text.
    - `filename`: Name the file as '{FirstName}_{LastName}_{DocumentType}.docx'
      where DocumentType is one of: CV, Resume, Cover_Letter
      e.g. 'John_Doe_Cover_Letter.docx', 'John_Doe_Resume.docx'
    """
    doc = Document()
    
    lines = text_content.split('\n')
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            doc.add_heading(stripped.replace("## ", ""), level=1)
        elif stripped.startswith("### "):
            doc.add_heading(stripped.replace("### ", ""), level=2)
        elif stripped.startswith("* ") or stripped.startswith("- "):
            doc.add_paragraph(stripped[2:], style='List Bullet')
        else:
            doc.add_paragraph(stripped)

    base, ext = os.path.splitext(filename)
    counter = 1
    final_filename = filename
    
    while True:
        try:
            doc.save(final_filename)
            break
        except PermissionError:
            final_filename = f"{base}_{counter}{ext}"
            counter += 1

    try:
        if platform.system() == "Windows":
            os.startfile(final_filename)
    except Exception:
        pass

    return f"Success! Document saved as '{final_filename}'."

# Wrap it for the agent
my_doc_tool = FunctionTool(save_to_word)

# CV Stylist Agent: Its job is to take the research findings, draft a CV in markdown, and 
# save it using the tool.
cv_stylist_agent = Agent(
    name="DocumentStylistAgent",
    model=my_model,
    instruction="""You are a professional career document formatter. You can create CVs, Resumes, and Cover Letters.

1. Identify the document type requested: CV, Resume, or Cover Letter.
2. Try to extract the person's full name from the research data.
   - If a clear full name is found: use Firstname_Lastname as the filename prefix.
   - If no name is found or it is unclear: use the filename prefix My instead e.g. My_CV.docx
3. Draft the complete, highly detailed Markdown document appropriate for the type:
   - CV: Full academic/professional history, all sections
   - Resume: Concise 1-2 page format, tailored to a role
   - Cover Letter: Professional letter format with date, recipient, body paragraphs, sign-off
4. Format the filename using the rules above:
   - With name: Firstname_Lastname_CV.docx
   - Without name: My_CV.docx, My_Resume.docx, My_Cover_Letter.docx
5. Execute save_to_word — pass your full markdown into text_content and the formatted filename into filename.
6. After the tool succeeds, return TWO things:
   - The FULL document markdown text so the user can read it
   - The exact filename used, in this exact format: Saved as My_CV.docx
7. If no background information was provided by the user at all, ask the user:
   "To generate your document, please share some background information such as your name, experience, skills, and education." Do NOT call save_to_word in this case.""",
    tools=[my_doc_tool]
)

# High-level orchestrator that routes between the sub-agents based on the user's request.
root_agent = Agent(
    # name="ResearchCoordinator",
    name="Edrick",
    model=my_model,
    instruction="""You are an expert in writting CVs, Resumes & Cover Letters and Doing Research.

Analyze the user's request and follow the correct path:

---
PATH 1: CAREER DOCUMENT GENERATION (CV, Resume, or Cover Letter)
Triggered when the user asks to create, write, or generate any of: a CV, resume, or cover letter.

Before calling any agent, check if the user has provided any personal background
such as their name, experience, skills, or education.

- If NO background info was provided: Do NOT call any agent. Instead ask the user:
  "To generate your document, I will need some background information.
   Please share details such as your full name, work experience, skills, and education."

- If background info WAS provided: proceed with the steps below:
  1. Call ResearchAgent to gather and structure the user's professional background.
  2. Pass that raw research AND the document type to DocumentStylistAgent.
  3. When DocumentStylistAgent returns, relay to the user:
     - The full document text
     - The confirmation of the filename saved e.g. Saved as Firstname_Lastname_Resume.docx
  DO NOT attempt to format, read, or save the document yourself.
---
PATH 2: GENERAL RESEARCH / SUMMARY
Triggered when the user asks for research, facts, or a summary on any topic.

1. Call ResearchAgent to gather findings.
2. Pass findings to SummarizerAgent for a concise bulleted summary.
3. Return the final summary to the user.
---
For anything else, respond helpfully from your own knowledge.
""",
    tools=[
        AgentTool(research_agent),
        AgentTool(cv_stylist_agent)
    ],
)

# Wrap your local CLI code in this guard
if __name__ == "__main__":
    # Run the orchestrator locally for rapid terminal debugging
    user_input = input("Enter your request: ")
    runner = InMemoryRunner(agent=root_agent)
    response = runner.run_debug(user_input)
    print(response)
