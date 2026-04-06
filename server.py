import os
import uuid
import hmac
import hashlib
import datetime
import asyncio
import logging
import httpx
import zipfile
import io
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from ci_cd_analyzer.observability import run_graph

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipelineiq")

app = FastAPI(title="PipelineIQ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GITHUB_TOKEN          = os.environ.get("GITHUB_TOKEN", "")
GITHUB_WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "")

runs_db: list[dict] = []


# ── HMAC verification ─────────────────────────────────────────────────────────

def verify_signature(body: bytes, sig_header: str | None) -> bool:
    if not GITHUB_WEBHOOK_SECRET:
        logger.warning("GITHUB_WEBHOOK_SECRET not set — skipping HMAC check")
        return True
    if not sig_header or not sig_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(
        GITHUB_WEBHOOK_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, sig_header)


# ── Log fetcher ───────────────────────────────────────────────────────────────

async def fetch_github_logs(repo: str, run_id: str) -> str:
    if not GITHUB_TOKEN:
        return f"[GITHUB_TOKEN not set — cannot fetch logs for run {run_id}]"

    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/logs"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept":        "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            r = await client.get(url, headers=headers)

        if r.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            lines: list[str] = []
            for name in sorted(z.namelist()):
                if name.endswith(".txt"):
                    lines.append(f"\n=== {name} ===")
                    lines.append(
                        z.open(name).read().decode("utf-8", errors="replace")
                    )
            return "\n".join(lines) or "[Empty log archive]"

        elif r.status_code == 410:
            return f"[Logs expired for run {run_id}]"
        else:
            return f"[Log fetch failed: HTTP {r.status_code}]"

    except Exception as e:
        logger.error("Log fetch error: %s", e)
        return f"[Log fetch exception: {e}]"


# ── Background analysis ───────────────────────────────────────────────────────

async def analyze_in_background(
    raw_log: str, metadata: dict, run_record: dict
) -> None:
    try:
        result = await run_graph(raw_log, metadata)
        report = result.get("final_report") or {
            "classification":  result.get("classification"),
            "severity":        result.get("severity"),
            "confidence":      result.get("confidence_score"),
            "root_cause":      result.get("root_cause"),
            "recommended_fix": result.get("recommended_fix"),
        }
        run_record.update({
            "report": report,
            "status": "completed",
            "completed_at": datetime.datetime.utcnow().isoformat(),
        })
        logger.info(
            "Done: %s → %s", metadata["run_id"], report.get("classification")
        )
    except Exception as e:
        logger.error("Analysis failed for %s: %s", metadata["run_id"], e)
        run_record.update({
            "status": "failed",
            "error":  str(e),
            "completed_at": datetime.datetime.utcnow().isoformat(),
        })


# ── GitHub webhook ────────────────────────────────────────────────────────────

@app.post("/webhook/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()

    # Verify HMAC
    if not verify_signature(body, request.headers.get("X-Hub-Signature-256")):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Only handle workflow_run events
    event = request.headers.get("X-GitHub-Event", "")
    if event != "workflow_run":
        return {"status": "ignored", "reason": f"event={event}"}

    payload    = await request.json()
    wf_run     = payload.get("workflow_run", {})
    action     = payload.get("action", "")
    conclusion = wf_run.get("conclusion", "")

    # Only failed completions
    if action != "completed" or conclusion != "failure":
        return {"status": "ignored", "reason": f"action={action} conclusion={conclusion}"}

    repo          = payload["repository"]["full_name"]
    github_run_id = str(wf_run["id"])
    branch        = wf_run.get("head_branch", "unknown")
    workflow_name = wf_run.get("name", "")
    run_id        = str(uuid.uuid4())

    metadata = {
        "repo":          repo,
        "branch":        branch,
        "stage":         workflow_name,
        "run_id":        run_id,
        "github_run_id": github_run_id,
        "timestamp":     datetime.datetime.utcnow().isoformat(),
        "team":          "",
    }

    logger.info("Webhook: failure on %s @ %s (gh_run=%s)", repo, branch, github_run_id)

    # Fetch the real logs
    raw_log = await fetch_github_logs(repo, github_run_id)

    # Create pending record — dashboard shows it immediately
    run_record = {
        "id":            run_id,
        "github_run_id": github_run_id,
        "timestamp":     metadata["timestamp"],
        "metadata":      metadata,
        "report":        None,
        "status":        "analyzing",
    }
    runs_db.append(run_record)

    # Dispatch — webhook returns in <200ms
    background_tasks.add_task(analyze_in_background, raw_log, metadata, run_record)

    return {"status": "accepted", "run_id": run_id}


# ── Manual endpoint (for testing without a real pipeline) ─────────────────────

class Metadata(BaseModel):
    repo:   str = "unknown/repo"
    branch: str = "main"
    stage:  str = ""
    team:   str = ""
    run_id: Optional[str] = None

class AnalysisRequest(BaseModel):
    log:      str
    metadata: Metadata