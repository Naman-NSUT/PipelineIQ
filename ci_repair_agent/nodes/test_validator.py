import json
import logging
import subprocess
from langchain_google_genai import ChatGoogleGenerativeAI
from ci_repair_agent.state import RepairState
from ci_repair_agent.config import config

logger = logging.getLogger("[TEST_VALIDATOR]")

def parse_json_response(content: str) -> dict:
    content = content.strip()
    if content.startswith("```json"):
        content = content.split("```json")[-1].rsplit("```", 1)[0].strip()
    elif content.startswith("```"):
        content = content.split("```")[-1].rsplit("```", 1)[0].strip()
    return json.loads(content)

def test_validator(state: RepairState) -> dict:
    logger.info("Running test suite via pytest...")
    
    result = subprocess.run("pytest", shell=True, capture_output=True, text=True, cwd=config.REPO_PATH)
    test_output = result.stdout + "\n" + result.stderr
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        api_key=config.GEMINI_API_KEY,
        temperature=0.1
    )
    
    prompt = f"""
You are an expert testing and validation system.
Analyze the test output and the repair report to validate the applied patches.

TEST OUTPUT:
{test_output}

REPAIR REPORT:
{json.dumps(state.get('repair_report', {}), indent=2)}

ERROR REPORT:
{json.dumps(state.get('error_report', {}), indent=2)}

Return a JSON object with:
- validation_status (str)
- tests_resolved (list of str)
- tests_still_failing (list of dict with test_name, reason, is_new_failure)
- regression_detected (bool)
- regression_details (str)
- patch_quality (str)
- patch_quality_notes (str)
- ready_to_push (bool)
- block_reason (str or null)

Return ONLY JSON, no markdown fences.
"""

    try:
        response = llm.invoke(prompt)
        try:
            parsed = parse_json_response(response.content)
        except Exception:
            response = llm.invoke(prompt + "\n\nCRITICAL: Return ONLY raw valid JSON, no markdown fences.")
            parsed = parse_json_response(response.content)
            
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", config.MAX_RETRIES)
        
        # Increment retry_count if not ready to push and under limit
        if not parsed.get("ready_to_push", False):
            if retry_count < max_retries:
                retry_count += 1
            else:
                # We exceeded retries, so we flag human review
                parsed["block_reason"] = "Exceeded max retries."
                
        return {
            "validation_report": parsed,
            "retry_count": retry_count
        }
    except Exception as e:
        logger.error(f"Test Validator failed: {e}")
        return {
            "requires_human_review": True,
            "human_review_reason": f"test_validator parse_error: {str(e)}"
        }
