import os
import asyncio
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

from agent import cicd_agent

# Load environment variables from .env file
load_dotenv()

# Render captures standard output, so keep runtime logs on the console.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("webhook_handler")

app = FastAPI(title="Self-Healing CI/CD Webhook Receiver")

class WebhookPayload(BaseModel):
    repo: str        # e.g., "owner/repo"
    run_id: str      # e.g., "123456789"
    branch: Optional[str] = None
    head_sha: Optional[str] = None
    logs: Optional[str] = None

async def fetch_and_log_github_failure(repository: str, run_id: str, head_sha: str = None, provided_logs: str = None):
    try:
        if provided_logs:
            logger.info("Using logs provided in webhook payload.")
            logger.info("=== START OF LOGS FOR JOB: test-frontend ===")
            print(provided_logs)
            logger.info("=== END OF LOGS FOR JOB: test-frontend ===")
        else:
            # Wait for the GitHub Action to finish executing and finalize logs
            logger.info(f"Waiting 15 seconds for GitHub Action run {run_id} to fully complete...")
            await asyncio.sleep(15)

            headers = {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28"
            }

            # Use environment token
            github_token = os.getenv("GITHUB_TOKEN")
            if github_token:
                headers["Authorization"] = f"Bearer {github_token}"
                logger.info("Using authorization token for GitHub API request.")
            else:
                logger.warning("No GitHub token provided. API requests might be rate-limited or fail for private repositories.")

            jobs_url = f"https://api.github.com/repos/{repository}/actions/runs/{run_id}/jobs"

            with httpx.Client(follow_redirects=True) as client:
                logger.info(f"Fetching jobs from URL: {jobs_url}")
                response = client.get(jobs_url, headers=headers)

                if response.status_code != 200:
                    logger.error(f"Failed to fetch jobs: HTTP {response.status_code} - {response.text}")
                    return

                jobs_data = response.json()
                jobs = jobs_data.get("jobs", [])
                logger.info(f"Found {len(jobs)} jobs in workflow run {run_id}")

                failed_jobs = [j for j in jobs if j.get("conclusion") == "failure" or j.get("status") == "completed" and j.get("conclusion") in ["failure", "cancelled"]]

                # Fallback to all jobs if no failed jobs are found, so triggering always prints logs
                jobs_to_log = failed_jobs if failed_jobs else jobs

                if not jobs_to_log:
                    logger.info("No jobs found in the workflow run.")
                    return

                for job in jobs_to_log:
                    job_id = job.get("id")
                    job_name = job.get("name")
                    logger.info(f"Processing failed job: {job_name} (ID: {job_id})")

                    logs_url = f"https://api.github.com/repos/{repository}/actions/jobs/{job_id}/logs"
                    logger.info(f"Fetching logs from: {logs_url}")

                    log_text = None
                    for attempt in range(1, 6):
                        logs_response = client.get(logs_url, headers=headers)
                        if logs_response.status_code == 200:
                            log_text = logs_response.text
                            break
                        else:
                            logger.warning(f"Failed to fetch logs for job {job_id} on attempt {attempt}/5: HTTP {logs_response.status_code}. Retrying in 5 seconds...")
                            await asyncio.sleep(5)

                    if log_text is None:
                        logger.error(f"Failed to fetch logs for job {job_id} after 5 attempts.")
                        continue

                    # Print the response logs directly to Vercel logs
                    logger.info(f"=== START OF LOGS FOR JOB: {job_name} ===")
                    print(log_text)
                    logger.info(f"=== END OF LOGS FOR JOB: {job_name} ===")

        if head_sha:
            logger.info("Triggering Google ADK Autonomous Agent for self-healing...")
            prompt = f"The pipeline failed for repo {repository} on run {run_id}. The base commit SHA for branching is {head_sha}. Please automatically execute your self-healing workflow to fix the bug."
            try:
                from google.adk.runners import InMemoryRunner
                runner = InMemoryRunner(agent=cicd_agent)
                response_events = await runner.run_debug(prompt)
                logger.info(f"Agent Execution Complete. Result: {response_events}")
            except Exception as e:
                logger.exception(f"Error executing agent: {e}")
        else:
            logger.warning("No head_sha provided in webhook payload. Cannot trigger self-healing agent.")

    except Exception as e:
        logger.exception(f"Error occurred during fetching GitHub Action response: {e}")

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

@app.post("/webhook")
async def receive_webhook(
    request: Request, payload: WebhookPayload, background_tasks: BackgroundTasks
):
    logger.info(f"🚀 WEBHOOK TRIGGERED! Pipeline failed for repository: {payload.repo}, Run ID: {payload.run_id}, Branch: {payload.branch}")

    auth = request.headers.get("Authorization")

    if not WEBHOOK_SECRET:
        logger.error("CRITICAL: WEBHOOK_SECRET is not set in the environment variables on this server!")
        raise HTTPException(
            status_code=503, detail="Webhook authentication is not configured"
        )

    if auth != f"Bearer {WEBHOOK_SECRET}":
        logger.warning(f"Unauthorized webhook attempt. Authentication failed.")
        raise HTTPException(status_code=401, detail="Unauthorized")

    logger.info("✅ Webhook authentication successful. Scheduling failure processing.")
    background_tasks.add_task(
        fetch_and_log_github_failure,
        payload.repo,
        payload.run_id,
        payload.head_sha,
        payload.logs
    )

    return {
        "status": "success",
        "message": f"Webhook received. Processing run {payload.run_id} in the background."
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}
