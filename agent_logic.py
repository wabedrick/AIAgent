import os
import platform
import itertools
from docx import Document
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from tavily import TavilyClient # type: ignore

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool, AgentTool
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

load_dotenv()

# ── 1. Model Definitions ──────────────────────────────────────────────────────

# Groq key rotation — cycles between two keys to double the free quota
groq_keys = [k for k in [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY_2"),
] if k]  # filters out None if second key isn't set

if not groq_keys:
    raise ValueError("At least one GROQ_API_KEY must be set in environment variables.")

groq_key_cycle = itertools.cycle(groq_keys)

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

def make_groq_model() -> LiteLlm:
    """Creates a fresh Groq model instance using the next key in rotation."""
    return LiteLlm(
        model="groq/llama-3.3-70b-versatile",
        api_key=next(groq_key_cycle),
        num_retries=5,
        timeout=120
    )

gemini_model = Gemini(
    model="gemini-2.0-flash",
    retry_options=retry_config
)

cerebras_model = LiteLlm(
    model="cerebras/llama-3.3-70b",
    api_key=os.getenv("CEREBRAS_API_KEY"),
    num_retries=5,
    timeout=120
)

# Start with Groq as the primary model
my_model = make_groq_model()

# ── 2. Search Tools ───────────────────────────────────────────────────────────

CURRENT_EVENT_KEYWORDS = [
    "today", "latest", "current", "recent", "now", "breaking",
    "this week", "this month", "this year", "2024", "2025", "2026",
    "news", "update", "happening", "live", "right now", "just",
    "yesterday", "tonight", "this morning", "new", "announced",
    "released", "launched", "trending"
]

def is_current_event_query(query: str) -> bool:
    """Detect if the query is about current or recent events."""
    return any(keyword in query.lower() for keyword in CURRENT_EVENT_KEYWORDS)

def search_with_duckduckgo(query: str) -> str:
    """Search using DuckDuckGo for general and historical queries."""
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=3)
        if not results:
            return "No results found."
        return "\n\n".join([
            f"Title: {res['title']}\nSnippet: {res['body']}"
            for res in results
        ])
    except Exception as e:
        return f"DuckDuckGo search failed: {str(e)}"

def search_with_tavily(query: str) -> str:
    """Search using Tavily for current events and real-time data."""
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=3,
            include_answer=True,
        )
        output = ""
        if response.get("answer"):
            output += f"Direct Answer: {response['answer']}\n\n"
        results = response.get("results", [])
        if results:
            output += "\n\n".join([
                f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content']}"
                for r in results
            ])
        return output if output else "No results found."
    except Exception as e:
        error_msg = str(e).lower()
        # Tavily quota exceeded — fall back to DuckDuckGo
        if any(term in error_msg for term in ["quota", "limit", "exceeded", "402", "429"]):
            print("[Search Router] Tavily quota exceeded, falling back to DuckDuckGo.", flush=True)
            return search_with_duckduckgo(query)
        return f"Tavily search failed: {str(e)}"

def web_search(query: str) -> str:
    """
    Smart search router:
    - Current/recent event queries → Tavily (real-time, accurate)
    - General/historical queries → DuckDuckGo (saves Tavily quota)
    - Tavily quota exceeded → automatically falls back to DuckDuckGo
    """
    if is_current_event_query(query):
        print(f"[Search Router] Current event → Tavily: {query}", flush=True)
        return search_with_tavily(query)
    else:
        print(f"[Search Router] General query → DuckDuckGo: {query}", flush=True)
        return search_with_duckduckgo(query)

my_search_tool = FunctionTool(web_search)

# ── 3. Document Tool ──────────────────────────────────────────────────────────

def save_to_word(text_content: str, filename: str = "Document.docx") -> str:
    """
    Creates a formatted Word document from markdown text.

    CRITICAL INSTRUCTION FOR LLM:
    - text_content: The FULL markdown text.
    - filename: Name the file using the person's actual name and document type:
      - With name: Firstname_Lastname_CV.docx
      - Without name: My_CV.docx, My_Resume.docx, My_Cover_Letter.docx
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

my_doc_tool = FunctionTool(save_to_word)

# ── 4. Agents ─────────────────────────────────────────────────────────────────

research_agent = Agent(
    name="ResearchAgent",
    model=my_model,
    instruction="""You are a specialized research agent. Your only job is to use the
    web_search tool to find 2-3 pieces of relevant information on the given topic and present the findings with citations.""",
    tools=[my_search_tool],
    output_key="research_findings",
)

summarizer_agent = Agent(
    name="SummarizerAgent",
    model=my_model,
    instruction="""Read the provided research findings: {research_findings}
Create a concise summary as a bulleted list with 3-5 key points.""",
    output_key="final_summary",
)

cv_stylist_agent = Agent(
    name="DocumentStylistAgent",
    model=my_model,
    instruction="""You are a professional career document formatter. You can create CVs, Resumes, and Cover Letters.

1. Identify the document type requested: CV, Resume, or Cover Letter.
2. Try to extract the person's full name from the research data.
   - If a clear full name is found: use Firstname_Lastname as the filename prefix.
   - If no name is found or it is unclear: use My as the filename prefix e.g. My_CV.docx
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
   - The exact filename used, in this exact format: Saved as Firstname_Lastname_CV.docx
7. If no background information was provided at all, ask:
   To generate your document, please share some background information such as your name, experience, skills, and education.
   Do NOT call save_to_word in this case.""",
    tools=[my_doc_tool]
)

root_agent = Agent(
    name="ResearchCoordinator",
    model=my_model,
    instruction="""You are a high-level routing orchestrator.

Analyze the user's request and follow the correct path:

---
PATH 1: CAREER DOCUMENT GENERATION (CV, Resume, or Cover Letter)
Triggered when the user asks to create, write, or generate any of: a CV, resume, or cover letter.

Before calling any agent, check if the user has provided any personal background
such as their name, experience, skills, or education.

- If NO background info was provided: Do NOT call any agent. Instead ask the user:
  To generate your document, I will need some background information.
  Please share details such as your full name, work experience, skills, and education.

- If background info WAS provided:
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

# ── 5. Model Switch Functions (must come after agents are defined) ─────────────

def switch_to_gemini():
    """Switch all agents to Gemini 2.0 Flash."""
    global my_model
    my_model = gemini_model
    print("[Model Router] Switching all agents → Gemini 2.0 Flash", flush=True)
    research_agent.model = gemini_model
    summarizer_agent.model = gemini_model
    cv_stylist_agent.model = gemini_model
    root_agent.model = gemini_model

def switch_to_cerebras():
    """Switch all agents to Cerebras llama-3.3-70b."""
    global my_model
    my_model = cerebras_model
    print("[Model Router] Switching all agents → Cerebras llama-3.3-70b", flush=True)
    research_agent.model = cerebras_model
    summarizer_agent.model = cerebras_model
    cv_stylist_agent.model = cerebras_model
    root_agent.model = cerebras_model

def switch_to_groq():
    """Switch all agents to a fresh Groq instance (rotates API keys)."""
    global my_model
    new_groq = make_groq_model()
    my_model = new_groq
    print("[Model Router] Switching all agents → Groq (key rotated)", flush=True)
    research_agent.model = new_groq
    summarizer_agent.model = new_groq
    cv_stylist_agent.model = new_groq
    root_agent.model = new_groq

# ── 6. Local CLI runner ───────────────────────────────────────────────────────

if __name__ == "__main__":
    user_input = input("Enter your request: ")
    runner = InMemoryRunner(agent=root_agent)
    response = runner.run_debug(user_input)
    print(response)