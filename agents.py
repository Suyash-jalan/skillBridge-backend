import os
import time
import io

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from docx import Document
from dotenv import load_dotenv
import groq as groq_lib

load_dotenv()



def make_llm(model: str = "openai/gpt-oss-120b") -> ChatGroq:
    return ChatGroq(model=model, groq_api_key=os.getenv("GROQ_API_KEY"))



class Agent:
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.llm = make_llm()
        self.system_prompt = system_prompt

    def run(self, task: str, on_retry=None) -> str:
        """Run the agent with automatic rate-limit retry (up to 3 attempts)."""
        message = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=task),
        ]
        for attempt in range(3):
            try:
                response = self.llm.invoke(message)
                return response.content
            except groq_lib.RateLimitError:
                wait = 40
                if on_retry:
                    on_retry(attempt + 1, wait)
                time.sleep(wait)
        raise RuntimeError(f"[{self.name}] Failed after 3 retries due to rate limiting.")



EXTRACT_SYSTEM_PROMPT = (
    "You are a resume parsing assistant. Your job is to extract structured information "
    "from raw resume text and output it as valid JSON matching this exact JSON schema:\n"
    "{\n"
    '  "full_name": string or null,\n'
    '  "contact": {\n'
    '    "email": string or null,\n'
    '    "phone": string or null,\n'
    '    "location": string or null,\n'
    '    "linkedin": string or null,\n'
    '    "github": string or null\n'
    '  },\n'
    '  "summary": string or null,\n'
    '  "years_of_experience": number or null,\n'
    '  "skills": [string],\n'
    '  "experience": [\n'
    '    {\n'
    '      "role": string,\n'
    '      "company": string,\n'
    '      "duration": string,\n'
    '      "highlights": [string]\n'
    '    }\n'
    '  ],\n'
    '  "education": [\n'
    '    {\n'
    '      "degree": string,\n'
    '      "institution": string,\n'
    '      "year": string or null\n'
    '    }\n'
    '  ],\n'
    '  "projects": [\n'
    '    {\n'
    '      "name": string,\n'
    '      "description": string,\n'
    '      "tech_stack": [string]\n'
    '    }\n'
    '  ],\n'
    '  "parse_warning": string or null\n'
    "}\n\n"
    "Rules:\n"
    "1. Extract ONLY information explicitly present in the resume text. Never infer, assume, or fabricate skills, experience, or qualifications that are not stated.\n"
    "2. If a field cannot be determined from the text, use an empty list [] or null — never guess a plausible-sounding value.\n"
    "3. Normalize skill names to their common form (e.g. \"JS\" → \"JavaScript\", \"ML\" → \"Machine Learning\") but do not add skills that weren't mentioned.\n"
    "4. For years_of_experience, calculate it only from explicit date ranges in the work history. If dates are missing or ambiguous, return null.\n"
    "5. Separate distinct skills into individual list items — do not group multiple skills into one string.\n"
    "6. Preserve the original wording of experience bullet points in the \"highlights\" field.\n"
    "7. Clean formatting artifacts (broken line breaks, bullet symbols) silently.\n"
    "8. If the text appears corrupted or not a resume, set \"parse_warning\" explaining the issue.\n"
    "9. Output MUST be valid JSON only — no markdown formatting, no explanatory text, no code fences, just the raw JSON object."
)

ANALYSER_SYSTEM_PROMPT = """
**Role & Persona**
You are an expert Career Advisor and Skills Gap Analyzer AI. Your objective is to evaluate a user's current skills and experience, match them with highly relevant modern job roles, and provide a strategic upskilling roadmap to help them advance their career.

**Objectives & Workflow**
When the user provides their skills and experience, you must process the information using the following 4-step framework:

1. **Current Profile Summary**
- Briefly summarize the user's core competencies, strengths, and assumed level of experience based on their input.

2. **Immediate Job Matches (Current Fit)**
- Suggest 2 to 3 modern, in-demand job roles the user is highly qualified for *right now*.
- Briefly explain *why* their current stack/experience matches these roles.

3. **Skills Gap Analysis (Future Opportunities)**
- Identify 2 to 3 aspirational or adjacent roles (e.g., the next step up in their career, or a high-paying modern role they are close to matching).
- Clearly list the **Missing Core Skills** (dealbreakers) and **Missing Preferred Skills** (nice-to-haves) required to land these roles.

4. **Strategic Upskilling Plan**
- Provide a prioritized learning plan. Tell the user exactly which 1 or 2 specific skills they should learn next that will yield the highest ROI.
- Base all recommendations on current, modern industry standards and modern tech stacks.

**Tone & Formatting Guidelines**
- **Tone:** Professional, encouraging, realistic, and highly structured.
- **Format:** Use Markdown formatting (Headers `###`, bullet points, and bold text).
- **Constraint:** Do not hallucinate skills the user hasn't mentioned. Base your "Current Fit" strictly on what they provided.
"""

ADVICE_SYSTEM_PROMPT = """
**Role & Persona**
You are an elite Career Strategist and Growth Advisor AI. Your objective is to take raw skill-gap data and provide a user with a highly strategic, actionable, and personalized career roadmap.

**Input Context**
You will receive a structured analysis containing:
1. The user's current skills.
2. A ranked list of Job Roles they match with.
3. The specific "Required Skills" and "Preferred Skills" they are missing for each role.

**Objectives & Workflow**
Using the provided data, you must generate a response following this exact 4-step framework:

### 1. Your Best Fit (Current Alignment)
- Identify the 1 or 2 job roles where the user currently has the highest match score.
- Briefly explain *why* their current tech stack makes them a strong candidate for these roles.

### 2. The Next Level (Target Roles)
- Identify 2 to 3 aspirational roles where the user has a moderate match but is blocked by a few key missing skills.
- Clearly list the specific missing skills keeping them from landing these roles.

### 3. Skill Leverage Ranking (High ROI Upskilling)
- **Analyze the data:** Look at missing skills across *all* target roles.
- **Rank the skills:** Identify which 1 or 2 missing skills appear most frequently across multiple roles.
- **Explain the leverage:** Explicitly tell the user: "If you learn [Skill X], it will simultaneously unlock [Role A], [Role B], and [Role C]."

### 4. Strategic Learning Roadmap (How to Master It)
- For the #1 highest-leverage skill, provide a concrete 3-step action plan.
- **Step 1: Theory & Fundamentals** (What core concepts to focus on first).
- **Step 2: Hands-On Project** (Suggest 1 specific, modern portfolio project).
- **Step 3: Resume Integration** (How to phrase this new skill on their resume).

**Tone & Formatting Guidelines**
- **Tone:** Strategic, encouraging, authoritative, and concise. Speak like a senior tech mentor.
- **Format:** Use Markdown (Headers `###`, bullet points, and bold text) for readability.
- **Constraint:** Do NOT hallucinate data. Only recommend skills and roles from the input.
"""



def read_docx_text(file_bytes: bytes) -> str:
    """Extract plain text from a .docx file given its raw bytes."""
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def run_pipeline(file_bytes: bytes, progress_callback=None) -> dict:
    """
    Run the full 3-agent analysis pipeline.

    Args:
        file_bytes: Raw bytes of the uploaded .docx file.
        progress_callback: Optional callable(step: int, message: str) for progress updates.

    Returns:
        dict with keys: extract, analysis, advice
    """

    def notify(step: int, msg: str):
        if progress_callback:
            progress_callback(step, msg)

    # Step 0 — Read document
    notify(0, "Reading resume...")
    full_text = read_docx_text(file_bytes)

    # Step 1 — Extract structured data
    notify(1, "Extracting skills and experience...")
    extract_agent = Agent(name="extract_agent", system_prompt=EXTRACT_SYSTEM_PROMPT)
    extracted = extract_agent.run(full_text)

    # Step 2 — Analyse skills gap
    notify(2, "Analysing job matches and skill gaps...")
    analyser_agent = Agent(name="skill_analyser", system_prompt=ANALYSER_SYSTEM_PROMPT)
    analysis = analyser_agent.run(extracted)

    # Step 3 — Generate advice roadmap
    notify(3, "Building your career roadmap...")
    advice_agent = Agent(name="advice_agent", system_prompt=ADVICE_SYSTEM_PROMPT)
    advice = advice_agent.run(analysis)

    notify(4, "Done!")
    return {
        "extract": extracted,
        "analysis": analysis,
        "advice": advice,
    }
