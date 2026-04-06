import json
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from ci_repair_agent.state import RepairState
from ci_repair_agent.config import config
from ci_repair_agent.tools.git_tools import apply_commands

logger = logging.getLogger("[GIT_PUSH]")

def parse_json_response(content: str) -> dict:
    content = content.strip()
    if content.startswith("```json"):
        content = content.split("```json")[-1].rsplit("```", 1)[0].strip()
    elif content.startswith("```"):
        content = content.split("```")[-1].rsplit("```", 1)[0].strip()
    return json.loads(content)

def git_push(state: RepairState) -> dict:
    logger.info("Generating Git push commands...")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        api_key=config.GEMINI_API_KEY,
        temperature=0.1
    )
    
    prompt = f"""
You are a senior DevOps agent.
Produce Git operations for a successfully validated repair. 

VALIDATION REPORT:
{json.dumps(state.get('validation_report', {}), indent=2)}

REPAIR REPORT:
{json.dumps(state.get('repair_report', {}), indent=2)}

ERROR REPORT:
{json.dumps(state.get('error_report', {}), indent=2)}

TRIGGER BRANCH:
{state.get('trigger_branch')}

CONFIG:
Base branch: {config.BASE_BRANCH}
Remote: {config.REMOTE_NAME}
Prefix: {config.COMMIT_PREFIX}

Return a JSON object with:
- target_branch (str)
- creates_new_branch (bool)
- new_branch_name (str - format: autofix/<kebab-case-issue-summary>)
- commands (list of str - exact shell strings to create branch, add files, commit, and push)
- commit_message (str)
- pr_title (str)
- pr_body (str - markdown)
- notify_team (bool)
- notification_reason (str or null)

Never push directly to main or master. Use shell strings carefully.
Return ONLY JSON, no markdown fences.
"""

    try:
        response = llm.invoke(prompt)
        try:
            parsed = parse_json_response(response.content)
        except Exception:
            response = llm.invoke(prompt + "\n\nCRITICAL: Return ONLY raw valid JSON, no markdown.")
            parsed = parse_json_response(response.content)
            
        cmds = parsed.get("commands", [])
        
        # Execute commands
        if cmds:
            try:
                apply_commands(cmds)
                logger.info("Successfully executed git commands.")
            except Exception as e:
                logger.error(f"Git application failed: {e}")
                parsed["git_execution_error"] = str(e)
                
        if parsed.get("notify_team"):
            print("\n" + "="*50)
            print(f"🚨 TEAM ALERT 🚨")
            print(f"Reason: {parsed.get('notification_reason')}")
            print("="*50 + "\n")
            
        return {"git_report": parsed}
    except Exception as e:
        logger.error(f"Git Push failed: {e}")
        return {
            "requires_human_review": True,
            "human_review_reason": f"git_push error: {str(e)}"
        }
