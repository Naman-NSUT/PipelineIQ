import json
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from ci_repair_agent.state import RepairState
from ci_repair_agent.config import config
from ci_repair_agent.tools.file_tools import write_patch

logger = logging.getLogger("[CODE_REPAIR]")

def parse_json_response(content: str) -> dict:
    content = content.strip()
    if content.startswith("```json"):
        content = content.split("```json")[-1].rsplit("```", 1)[0].strip()
    elif content.startswith("```"):
        content = content.split("```")[-1].rsplit("```", 1)[0].strip()
    return json.loads(content)

def code_repair(state: RepairState) -> dict:
    logger.info("Starting code repair logic...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        api_key=config.GEMINI_API_KEY,
        temperature=0.1
    )
    
    prompt = f"""
You are a senior backend engineer specializing in API service repair.
You fix broken code based on a structured error report.

INPUTS:
- error_report: {json.dumps(state.get('error_report', {}), indent=2)}
- source_files: {json.dumps(state.get('source_files', {}), indent=2)}
- test_files: {json.dumps(state.get('test_files', {}), indent=2)}

YOUR TASK:
Return a JSON object in this exact structure:
{{
  "repair_summary": "<What was broken and what you changed, in 2-3 sentences>",
  "confidence": "<high | medium | low>",
  "patches": [
    {{
      "file": "<filepath>",
      "action": "<modify | create | delete>",
      "original_snippet": "<the exact lines being replaced, or null for create>",
      "patched_snippet": "<the replacement lines>",
      "reason": "<why this change fixes the issue>"
    }}
  ],
  "side_effects": ["<any potential side effects>"],
  "requires_dependency_change": <true | false>,
  "dependency_changes": "<description if true, else null>"
}}

RULES:
- Only modify files listed in error_report.affected_files.
- Never modify test files.
- Never change signatures/APIs unless requested.
- Patches must be minimal replacing lines exactly.
- Output ONLY the JSON object. No markdown.
"""

    try:
        response = llm.invoke(prompt)
        try:
            parsed = parse_json_response(response.content)
        except Exception:
            logger.warning("Parsing failed, retrying once...")
            response = llm.invoke(prompt + "\n\nCRITICAL: Return ONLY raw valid JSON, no markdown fences.")
            parsed = parse_json_response(response.content)
            
        req_review = state.get("requires_human_review", False)
        
        if parsed.get("confidence") == "low":
            req_review = True
            
        # Apply patches
        for patch in parsed.get("patches", []):
            try:
                write_patch(patch["file"], patch.get("original_snippet"), patch.get("patched_snippet"))
                logger.info(f"Applied patch to {patch['file']}")
            except Exception as e:
                logger.error(f"Failed to apply patch to {patch['file']}: {e}")
                req_review = True
        
        return {
            "repair_report": parsed,
            "requires_human_review": req_review
        }
    except Exception as e:
        logger.error(f"Code Repair failed: {e}")
        return {
            "requires_human_review": True,
            "human_review_reason": f"parse_error: {str(e)}"
        }
