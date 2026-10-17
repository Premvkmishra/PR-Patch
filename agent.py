import base64
import logging
import os
from typing import AsyncGenerator
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

load_dotenv()
logger = logging.getLogger(__name__)


def _headers() -> dict[str, str]:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not configured")
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _tool_result(name: str, arguments: str, success: bool, detail: str = "") -> None:
    result = "OK" if success else f"FAIL {detail}"
    logger.info("AGENT ACTION: %s %s result=%s", name, arguments, result)


def _redact_secrets(text: str) -> str:
    for variable in ("GITHUB_TOKEN", "WEBHOOK_SECRET", "GROQ_API_KEY", "GOOGLE_API_KEY"):
        secret = os.getenv(variable)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


class GeminiWithGroqFallback(BaseLlm):
    model: str = "gemini-2.5-flash"
    groq_model: str = "groq/llama-3.3-70b-versatile"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        gemini = Gemini(model=self.model)
        received_response = False
        failure: Exception | None = None

        try:
            async for response in gemini.generate_content_async(
                llm_request, stream=stream
            ):
                if response.error_code:
                    raise RuntimeError(
                        f"Gemini returned an error: {response.error_message or response.error_code}"
                    )
                if response.content is None or not response.content.parts:
                    raise RuntimeError("Gemini returned an empty response.")
                received_response = True
                yield response
        except Exception as error:
            if received_response:
                raise
            failure = error
        else:
            if received_response:
                return
            failure = RuntimeError("Gemini returned no response.")

        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError(
                "Gemini failed and GROQ_API_KEY is not configured."
            ) from failure

        logger.warning(
            "[MODEL] Gemini failed before responding; trying Groq fallback: %s",
            failure,
        )
        from google.adk.models.lite_llm import LiteLlm

        groq = LiteLlm(model=self.groq_model)
        async for response in groq.generate_content_async(
            llm_request, stream=stream
        ):
            yield response


def fetch_run_jobs(repo: str, run_id: str) -> str:
    logger.info("AGENT ACTION: fetch_run_jobs repo=%s run=%s start", repo, run_id)
    try:
        headers = _headers()
    except RuntimeError as error:
        _tool_result("fetch_run_jobs", f"repo={repo} run={run_id}", False, str(error))
        return f"Error fetching jobs: {error}"

    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs"
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        response = client.get(url, headers=headers)
        if response.status_code != 200:
            _tool_result(
                "fetch_run_jobs",
                f"repo={repo} run={run_id}",
                False,
                f"HTTP {response.status_code}",
            )
            return f"Error fetching jobs: HTTP {response.status_code} - {response.text}"

        jobs = response.json().get("jobs", [])
        failed_jobs = [
            job for job in jobs if job.get("conclusion") == "failure"
        ]
        if not failed_jobs:
            _tool_result(
                "fetch_run_jobs", f"repo={repo} run={run_id}", False, "no failed jobs"
            )
            return f"No failed jobs found for run {run_id}."

        job_logs = []
        for job in failed_jobs:
            job_id = job.get("id")
            job_name = job.get("name", str(job_id))
            if not job_id:
                _tool_result(
                    "fetch_job_logs",
                    f"repo={repo} job={job_name!r}",
                    False,
                    "missing job id",
                )
                continue
            logs_response = client.get(
                f"https://api.github.com/repos/{repo}/actions/jobs/{job_id}/logs",
                headers=headers,
            )
            if logs_response.status_code == 200:
                logger.info(
                    "AGENT ACTION: fetch_job_logs repo=%s job=%r result=OK",
                    repo,
                    job_name,
                )
                job_logs.append(
                    f"=== Logs for job: {job_name} ===\n{logs_response.text}"
                )
            else:
                _tool_result(
                    "fetch_job_logs",
                    f"repo={repo} job={job_name!r}",
                    False,
                    f"HTTP {logs_response.status_code}",
                )

        result = _redact_secrets("\n\n".join(job_logs))
        _tool_result(
            "fetch_run_jobs",
            f"repo={repo} run={run_id}",
            bool(result),
            "" if result else "failure logs unavailable",
        )
        return result or f"No logs available for run {run_id}."


def get_file_contents(repo: str, path: str, ref: str = "") -> str:
    logger.info(
        "AGENT ACTION: get_file_contents repo=%s path=%s ref=%s start",
        repo,
        path,
        ref or "default",
    )
    try:
        headers = _headers()
    except RuntimeError as error:
        _tool_result("get_file_contents", f"path={path} ref={ref}", False, str(error))
        return f"Error fetching file: {error}"

    url = f"https://api.github.com/repos/{repo}/contents/{quote(path, safe='/')}"
    params = {"ref": ref} if ref else None
    with httpx.Client(timeout=30) as client:
        response = client.get(url, headers=headers, params=params)
        if response.status_code != 200:
            _tool_result(
                "get_file_contents",
                f"path={path} ref={ref or 'default'}",
                False,
                f"HTTP {response.status_code}",
            )
            return f"Error fetching file: HTTP {response.status_code}"
        data = response.json()
        if "content" not in data:
            _tool_result(
                "get_file_contents",
                f"path={path} ref={ref or 'default'}",
                False,
                "content missing",
            )
            return "Error fetching file: GitHub response did not include content."
        content = base64.b64decode(data["content"]).decode("utf-8")
        _tool_result(
            "get_file_contents",
            f"path={path} ref={ref or 'default'}",
            True,
        )
        return content


def create_branch(
    repo: str, base_sha: str, new_branch_name: str, run_id: str = ""
) -> str:
    logger.info(
        "AGENT ACTION: create_branch repo=%s branch=%s base=%s start",
        repo,
        new_branch_name,
        base_sha[:7],
    )
    if (
        not new_branch_name.startswith("prpatch/")
        or len(base_sha) < 7
        or base_sha[:7] not in new_branch_name
        or (run_id and run_id not in new_branch_name)
    ):
        _tool_result(
            "create_branch",
            f"branch={new_branch_name}",
            False,
            "must start prpatch/, include base SHA prefix, and include run ID",
        )
        return (
            "Error creating branch: name must start with prpatch/, include "
            "the first 7 characters of base_sha, and include run_id."
        )
    try:
        headers = _headers()
    except RuntimeError as error:
        _tool_result("create_branch", f"branch={new_branch_name}", False, str(error))
        return f"Error creating branch: {error}"

    url = f"https://api.github.com/repos/{repo}/git/refs"
    payload = {"ref": f"refs/heads/{new_branch_name}", "sha": base_sha}
    with httpx.Client(timeout=30) as client:
        response = client.post(url, headers=headers, json=payload)
        if response.status_code == 201:
            logger.info(
                "AGENT ACTION: Branch created: %s  (base: %s)",
                new_branch_name,
                base_sha[:7],
            )
            _tool_result("create_branch", f"branch={new_branch_name}", True)
            return f"Successfully created branch {new_branch_name}"
        _tool_result(
            "create_branch",
            f"branch={new_branch_name}",
            False,
            f"HTTP {response.status_code}",
        )
        return f"Error creating branch: HTTP {response.status_code} - {response.text}"


def update_file(repo: str, path: str, content: str, branch: str, sha: str) -> str:
    logger.info(
        "AGENT ACTION: update_file repo=%s path=%s branch=%s start",
        repo,
        path,
        branch,
    )
    try:
        headers = _headers()
    except RuntimeError as error:
        _tool_result("update_file", f"path={path} branch={branch}", False, str(error))
        return f"Error committing file: {error}"

    url = f"https://api.github.com/repos/{repo}/contents/{quote(path, safe='/')}"
    payload = {
        "message": "AI Auto-Fix: Resolved CI/CD Pipeline Failure",
        "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
        "sha": sha,
        "branch": branch,
    }
    with httpx.Client(timeout=30) as client:
        response = client.put(url, headers=headers, json=payload)
        if response.status_code in (200, 201):
            _tool_result("update_file", f"path={path} branch={branch}", True)
            return "Successfully committed fix!"
        _tool_result(
            "update_file",
            f"path={path} branch={branch}",
            False,
            f"HTTP {response.status_code}",
        )
        return f"Error committing file: HTTP {response.status_code} - {response.text}"


def get_file_sha(repo: str, path: str, ref: str = "") -> str:
    logger.info(
        "AGENT ACTION: get_file_sha repo=%s path=%s ref=%s start",
        repo,
        path,
        ref or "default",
    )
    try:
        headers = _headers()
    except RuntimeError as error:
        _tool_result("get_file_sha", f"path={path} ref={ref}", False, str(error))
        return f"Error fetching file SHA: {error}"

    url = f"https://api.github.com/repos/{repo}/contents/{quote(path, safe='/')}"
    params = {"ref": ref} if ref else None
    with httpx.Client(timeout=30) as client:
        response = client.get(url, headers=headers, params=params)
        if response.status_code != 200:
            _tool_result(
                "get_file_sha",
                f"path={path} ref={ref or 'default'}",
                False,
                f"HTTP {response.status_code}",
            )
            return f"Error fetching file SHA: HTTP {response.status_code}"
        sha = response.json().get("sha")
        if not sha:
            _tool_result(
                "get_file_sha",
                f"path={path} ref={ref or 'default'}",
                False,
                "sha missing",
            )
            return "Error fetching file SHA: GitHub response did not include a SHA."
        _tool_result(
            "get_file_sha",
            f"path={path} ref={ref or 'default'}",
            True,
        )
        return sha


def create_pull_request(repo: str, branch: str, base: str, title: str, body: str) -> str:
    logger.info(
        "AGENT ACTION: create_pull_request repo=%s head=%s base=%s start",
        repo,
        branch,
        base,
    )
    try:
        headers = _headers()
    except RuntimeError as error:
        _tool_result(
            "create_pull_request",
            f"head={branch} base={base}",
            False,
            str(error),
        )
        return f"Error creating Pull Request: {error}"

    url = f"https://api.github.com/repos/{repo}/pulls"
    payload = {"title": title, "head": branch, "base": base, "body": body}
    with httpx.Client(timeout=30) as client:
        response = client.post(url, headers=headers, json=payload)
        if response.status_code == 201:
            pr_url = response.json().get("html_url")
            if pr_url:
                logger.info(
                    "AGENT ACTION: Pull Request opened: %s  (%s -> %s)",
                    pr_url,
                    branch,
                    base,
                )
            _tool_result("create_pull_request", f"head={branch} base={base}", bool(pr_url), str(pr_url or "URL missing"))
            return f"Successfully created Pull Request: {pr_url}"
        _tool_result(
            "create_pull_request",
            f"head={branch} base={base}",
            False,
            f"HTTP {response.status_code}",
        )
        return (
            f"Error creating Pull Request: HTTP {response.status_code} - "
            f"{response.text}"
        )


def list_repository_files(repo: str, branch: str = "main") -> str:
    logger.info("AGENT ACTION: list_repository_files repo=%s ref=%s start", repo, branch)
    try:
        headers = _headers()
    except RuntimeError as error:
        _tool_result("list_repository_files", f"ref={branch}", False, str(error))
        return f"Error listing repository files: {error}"

    ref = quote(branch, safe="")
    url = f"https://api.github.com/repos/{repo}/git/trees/{ref}?recursive=1"
    with httpx.Client(timeout=30) as client:
        response = client.get(url, headers=headers)
        if response.status_code != 200:
            _tool_result(
                "list_repository_files",
                f"ref={branch}",
                False,
                f"HTTP {response.status_code}",
            )
            return (
                f"Error listing repository files: HTTP {response.status_code} - "
                f"{response.text}"
            )
        tree = response.json().get("tree", [])
        file_paths = [
            item.get("path") for item in tree if item.get("type") == "blob"
        ]
        _tool_result(
            "list_repository_files",
            f"ref={branch}",
            True,
            f"{len(file_paths)} files",
        )
        return "\n".join(file_paths)


cicd_agent = Agent(
    name="cicd_self_healing_agent",
    model=GeminiWithGroqFallback(),
    instruction="""
You are a CI/CD self-healing agent. The user prompt gives repo, run_id,
FAILING_BRANCH, FAILING_SHA, and relevant failure logs.

Follow these steps:
1. Call list_repository_files(repo, FAILING_SHA) to list files at the exact
   failing commit.
2. Call fetch_run_jobs(repo, run_id), then identify the failed check, root
   cause, and the source file that caused it. Fix source code, not tests,
   unless a test is clearly wrong.
3. Read the broken source and any needed dependencies with
   get_file_contents(repo, path, FAILING_SHA).
4. Make a minimal fix. Describe the failing check, root cause, change, and
   modified file in the pull-request body, and state it was generated by Prpatch.
5. Create a unique branch named
   prpatch/fix-<short-description>-<first-7-of-FAILING_SHA>-<run_id>.
   It must begin with prpatch/ and include both the SHA prefix and run ID.
6. Call create_branch(repo, FAILING_SHA, branch_name, run_id).
7. Call get_file_sha(repo, fixed_file, FAILING_SHA), then
   update_file(repo, fixed_file, fixed_content, branch_name, file_sha).
8. Call create_pull_request(repo, branch_name, FAILING_BRANCH,
   "Prpatch: <short description>", body). The base must be FAILING_BRANCH,
   never hardcode main.

Stop and explain clearly if any tool reports an error. Do not claim a pull
request was created unless create_pull_request returns its URL.
""",
    tools=[
        fetch_run_jobs,
        get_file_contents,
        create_branch,
        update_file,
        get_file_sha,
        create_pull_request,
        list_repository_files,
    ],
)
