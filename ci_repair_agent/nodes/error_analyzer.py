import json
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from ci_repair_agent.state import RepairState
from ci_repair_agent.config import config

logger = logging.getLogger("[ERROR_ANALYZER]")

def parse_json_response(content: str) -> dict:
    content = content.strip()
    if content.startswith("```json"):
        content = content.split("```json")[-1].rsplit("```", 1)[0].strip()
    elif content.startswith("```"):
        content = content.split("```")[-1].rsplit("```", 1)[0].strip()
    return json.loads(content)

def error_analyzer(state: RepairState) -> dict:
    logger.info("Analyzing CI logs...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        api_key=config.GEMINI_API_KEY,
        temperature=0.1
    )
    
    prompt = f"""
You are an expert CI/CD error analyzer.
Analyze the following CI failure and return ONLY a JSON object with these fields:
- failure_type (str)
- severity (str)
- root_cause_summary (str)
- affected_files (list of str)
- error_details (dict with file, line, error_message, stack_trace)
- failed_tests (list of dicts with test_name, expected, actual, file)
- suggested_fix_strategy (str)
- requires_human_review (bool)
- human_review_reason (str or null)

CI Logs:
{state.get('ci_logs', '')}

Test Report:
{state.get('test_report', '')}

File Tree:
{state.get('file_tree', [])}

Trigger Branch:
{state.get('trigger_branch', '')}

Return ONLY the JSON object. No markdown, no explanation outside the JSON.
"""
    
    try:
        response = llm.invoke(prompt)
        try:
            parsed = parse_json_response(response.content)
        except Exception:
            # Retry once
            logger.warning("Failed to parse JSON, retrying once...")
            response = llm.invoke(prompt + "\n\nCRITICAL ERROR: You must return ONLY raw valid JSON, without any markdown formatting like ```json")
            parsed = parse_json_response(response.content)
            
        req_review = parsed.get("requires_human_review", False)
        return {
            "error_report": parsed,
            "requires_human_review": req_review
        }
    except Exception as e:
        logger.error(f"Error Analyzer failed: {e}")
        return {
            "requires_human_review": True,
            "human_review_reason": f"parse_error: {str(e)}"
        }
