import asyncio
import hmac
import inspect
import logging
import os
import re
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel

from agent import cicd_agent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("webhook_handler")
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI(title="Self-Healing CI/CD Webhook Receiver")


class WebhookPayload(BaseModel):
    repo: str
    run_id: str
    branch: Optional[str] = None
    head_sha: Optional[str] = None
    target_branch: Optional[str] = None
    logs: Optional[str] = None


def _last_lines(log_text: str, limit: int = 100) -> str:
    for variable in (
        "GITHUB_TOKEN",
        "WEBHOOK_SECRET",
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
    ):
        secret = os.getenv(variable)
        if secret:
            log_text = log_text.replace(secret, "[REDACTED]")
    lines = log_text.splitlines()
    return "\n".join(lines[-limit:])


def _failed_test_count(log_text: str) -> Optional[int]:
    matches = re.findall(r"(?m)^=+\s+(\d+)\s+failed\b", log_text)
    return int(matches[-1]) if matches else None


def _event_text(events) -> str:
    event_list = events if isinstance(events, (list, tuple)) else [events]
    text_parts = []
    for event in event_list:
        content = getattr(event, "content", None)
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                text_parts.append(text)
    return "\n".join(text_parts)


async def run_agent(agent, prompt: str, user_id: str = "ci"):
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    app_name = "prpatch"
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session = runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id,
    )
    if inspect.isawaitable(session):
        session = await session

    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    events = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=message,
    ):
        events.append(event)
    return events


async def fetch_and_log_github_failure(
    repository: str,
    run_id: str,
    head_sha: Optional[str] = None,
    provided_logs: Optional[str] = None,
    target_branch: Optional[str] = None,
):
    failing_branch = target_branch or "main"
    try:
        if not head_sha:
            logger.error(
                "[SUMMARY] FAILURE repo=%s run=%s reason=missing head_sha",
                repository,
                run_id,
            )
            return

        token = os.getenv("GITHUB_TOKEN")
        if not token:
            logger.error(
                "[SUMMARY] FAILURE repo=%s run=%s reason=GITHUB_TOKEN is not configured",
                repository,
                run_id,
            )
            return

        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Authorization": f"Bearer {token}",
        }
        logger.info(
            "[3/7] 📥 Fetching failure logs  repo=%s run=%s",
            repository,
            run_id,
        )
        combined_logs = provided_logs or ""

        with httpx.Client(follow_redirects=True, timeout=30) as client:
            jobs_url = (
                f"https://api.github.com/repos/{repository}"
                f"/actions/runs/{run_id}/jobs"
            )
            jobs_response = client.get(jobs_url, headers=headers)
            if jobs_response.status_code != 200:
                logger.error(
                    "[SUMMARY] FAILURE repo=%s run=%s reason=GitHub jobs API HTTP %s",
                    repository,
                    run_id,
                    jobs_response.status_code,
                )
                return

            failed_jobs = [
                job
                for job in jobs_response.json().get("jobs", [])
                if job.get("conclusion") == "failure"
            ]
            if not failed_jobs:
                logger.error(
                    "[SUMMARY] FAILURE repo=%s run=%s reason=no failed jobs found",
                    repository,
                    run_id,
                )
                return

            for job in failed_jobs:
                job_id = job.get("id")
                job_name = job.get("name", str(job_id))
                if not job_id:
                    logger.error(
                        "AGENT ACTION: fetch_job_logs job=%r result=FAILURE reason=missing id",
                        job_name,
                    )
                    continue

                log_response = None
                for attempt in range(3):
                    log_response = client.get(
                        f"https://api.github.com/repos/{repository}"
                        f"/actions/jobs/{job_id}/logs",
                        headers=headers,
                    )
                    if log_response.status_code == 200:
                        break
                    if attempt < 2:
                        await asyncio.sleep(2)

                if log_response is None or log_response.status_code != 200:
                    status_code = log_response.status_code if log_response else "no response"
                    logger.error(
                        "AGENT ACTION: fetch_job_logs job=%r result=FAILURE http_status=%s",
                        job_name,
                        status_code,
                    )
                    continue

                job_log = log_response.text
                combined_logs += f"\n=== Failed job: {job_name} ===\n{job_log}"
                test_count = _failed_test_count(job_log)
                count_note = f"  ({test_count} tests failed)" if test_count is not None else ""
                logger.info(
                    "[3/7] 📥 Fetched failure logs for job %r%s",
                    job_name,
                    count_note,
                )
                logger.info("[LOGS] job=%r last_lines=100\n%s", job_name, _last_lines(job_log))

        if not combined_logs:
            logger.error(
                "[SUMMARY] FAILURE repo=%s run=%s reason=no failure logs available",
                repository,
                run_id,
            )
            return

        if provided_logs:
            logger.info("[3/7] 📥 Using failure logs supplied in webhook payload")
            logger.info("[LOGS] last_lines=100\n%s", _last_lines(provided_logs))

        logger.info("[4/7] 🤖 Agent started  model=gemini-2.5-flash")
        prompt = (
            f"repo={repository}\nrun_id={run_id}\n"
            f"FAILING_BRANCH={failing_branch}\nFAILING_SHA={head_sha}\n"
            f"The failing branch is {failing_branch}. Read files from this "
            f"branch at failing commit {head_sha}, and open the fix Pull "
            f"Request into {failing_branch}. Include the failing check, "
            "root cause, change made, modified files, and that the fix was "
            f"generated by Prpatch.\nFailure logs:\n{_last_lines(combined_logs)}"
        )

        events = await run_agent(cicd_agent, prompt)
        response_text = _event_text(events)
        all_event_text = "\n".join((response_text, str(events)))
        if response_text:
            logger.info("[5/7] 🔎 Agent summary: %s", response_text[-500:].replace("\n", " "))
        branch_matches = re.findall(r"prpatch/[A-Za-z0-9._/-]+", all_event_text)
        pull_requests = re.findall(
            r"https://github\.com/[^\s\"'<>]+/pull/\d+",
            all_event_text,
        )
        if pull_requests:
            pr_url = pull_requests[-1].rstrip(".,)")
            if branch_matches:
                logger.info(
                    "[6/7] 🌿 Repair branch: %s  (base: %s)",
                    branch_matches[-1].rstrip(".,)"),
                    failing_branch,
                )
            logger.info(
                "[7/7] ✅ Pull Request opened: %s  (prpatch/... -> %s)",
                pr_url,
                failing_branch,
            )
            logger.info(
                "[SUMMARY] SUCCESS repo=%s run=%s pr=%s",
                repository,
                run_id,
                pr_url,
            )
        else:
            logger.error(
                "[SUMMARY] FAILURE repo=%s run=%s reason=agent completed without a Pull Request; inspect tool results",
                repository,
                run_id,
            )
    except Exception as error:
        logger.exception(
            "[SUMMARY] FAILURE repo=%s run=%s reason=%s",
            repository,
            run_id,
            error,
        )


WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")


@app.post("/webhook")
async def receive_webhook(
    request: Request,
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
):
    failing_branch = payload.target_branch or "main"
    logger.info(
        "[1/7] 🔔 Webhook received  repo=%s run=%s branch=%s",
        payload.repo,
        payload.run_id,
        failing_branch,
    )

    if not WEBHOOK_SECRET:
        logger.error(
            "[SUMMARY] FAILURE repo=%s run=%s reason=WEBHOOK_SECRET is not configured",
            payload.repo,
            payload.run_id,
        )
        raise HTTPException(
            status_code=503,
            detail="Webhook authentication is not configured",
        )

    auth = request.headers.get("Authorization")
    if not auth or not hmac.compare_digest(auth, f"Bearer {WEBHOOK_SECRET}"):
        logger.warning(
            "[SUMMARY] FAILURE repo=%s run=%s reason=unauthorized webhook",
            payload.repo,
            payload.run_id,
        )
        raise HTTPException(status_code=401, detail="Unauthorized")

    logger.info("[2/7] 🔐 Auth OK")
    background_tasks.add_task(
        fetch_and_log_github_failure,
        payload.repo,
        payload.run_id,
        payload.head_sha,
        payload.logs,
        failing_branch,
    )
    return {
        "status": "success",
        "message": f"Webhook received. Processing run {payload.run_id} in the background.",
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}
