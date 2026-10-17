import os
import httpx
import base64
import logging
from typing import AsyncGenerator
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

load_dotenv()
logger = logging.getLogger(__name__)


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
            "Gemini failed before returning a response; trying the Groq fallback: %s",
            failure,
        )
        from google.adk.models.lite_llm import LiteLlm

        groq = LiteLlm(model=self.groq_model)
        async for response in groq.generate_content_async(
            llm_request, stream=stream
        ):
            yield response

# Tool 1: fetch_run_jobs
def fetch_run_jobs(repo: str, run_id: str) -> str:
    """
    Fetches failed job logs for a specific GitHub Actions workflow run.
    """
    print(f"AGENT ACTION: Fetching jobs for run {run_id} to read logs...")
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token: headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs"
    with httpx.Client(follow_redirects=True) as client:
        response = client.get(url, headers=headers)
        if response.status_code != 200:
            return f"Error fetching jobs: HTTP {response.status_code} - {response.text}"

        jobs = response.json().get("jobs", [])
        failed_jobs = [
            job for job in jobs
            if job.get("conclusion") in ("failure", "cancelled")
        ]
        if not failed_jobs:
            return f"No failed jobs found for run {run_id}. Job details: {response.text}"

        job_logs = []
        for job in failed_jobs:
            job_id = job.get("id")
            if not job_id:
                continue
            logs_url = f"https://api.github.com/repos/{repo}/actions/jobs/{job_id}/logs"
            logs_response = client.get(logs_url, headers=headers)
            if logs_response.status_code == 200:
                job_logs.append(
                    f"=== Logs for job: {job.get('name', job_id)} ===\n"
                    f"{logs_response.text}"
                )
            else:
                job_logs.append(
                    f"Could not fetch logs for job {job.get('name', job_id)}: "
                    f"HTTP {logs_response.status_code} - {logs_response.text}"
                )
        return "\n\n".join(job_logs) or f"No logs available for run {run_id}."

# Tool 2: get_file_contents
def get_file_contents(repo: str, path: str) -> str:
    """
    Fetches the actual source code of a file from the repository.
    Returns the decoded file content as a string.
    """
    print(f"AGENT ACTION: Fetching source code for {path}...")
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token: headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    with httpx.Client() as client:
        response = client.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "content" in data:
                return base64.b64decode(data["content"]).decode('utf-8')
            return "No content field returned."
        return f"Error fetching file: HTTP {response.status_code}"

# Tool 3: create_branch
def create_branch(repo: str, base_sha: str, new_branch_name: str) -> str:
    """
    Creates a new git branch pointing to the base_sha.
    Returns a success or failure message.
    """
    print(f"AGENT ACTION: Creating new branch {new_branch_name}...")
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token: headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com/repos/{repo}/git/refs"
    payload = {"ref": f"refs/heads/{new_branch_name}", "sha": base_sha}

    with httpx.Client() as client:
        response = client.post(url, headers=headers, json=payload)
        if response.status_code == 201 or (response.status_code == 422 and "Reference already exists" in response.text):
            return f"Successfully created branch {new_branch_name}"
        return f"Error creating branch: HTTP {response.status_code} - {response.text}"

# Tool 4: update_file
def update_file(repo: str, path: str, content: str, branch: str, sha: str) -> str:
    """
    Commits the fixed code (content) to the specified file path on the specified branch.
    You must provide the blob sha of the file being replaced.
    Returns a success or failure message.
    """
    print(f"AGENT ACTION: Committing fixed code for {path} to branch {branch}...")
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token: headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
    payload = {
        "message": "AI Auto-Fix: Resolved CI/CD Pipeline Failure",
        "content": encoded_content,
        "sha": sha,
        "branch": branch
    }
    with httpx.Client() as client:
        response = client.put(url, headers=headers, json=payload)
        if response.status_code in [200, 201]:
            return "Successfully committed fix!"
        return f"Error committing file: HTTP {response.status_code} - {response.text}"

# Tool 5: get_file_sha
def get_file_sha(repo: str, path: str) -> str:
    """
    Gets the blob SHA for a specific file in the repository. This is required before calling update_file.
    """
    print(f"AGENT ACTION: Fetching blob SHA for {path}...")
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token: headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    with httpx.Client() as client:
        response = client.get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get("sha", "")
        return ""

# Tool 6: create_pull_request
def create_pull_request(repo: str, branch: str, base: str, title: str, body: str) -> str:
    """
    Creates a new Pull Request on GitHub from the specified branch to the base branch.
    Returns the URL of the created PR or failure message.
    """
    print(f"AGENT ACTION: Creating Pull Request from {branch} to {base}...")
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token: headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com/repos/{repo}/pulls"
    payload = {
        "title": title,
        "head": branch,
        "base": base,
        "body": body
    }
    with httpx.Client() as client:
        response = client.post(url, headers=headers, json=payload)
        if response.status_code == 201:
            pr_url = response.json().get("html_url")
            return f"Successfully created Pull Request: {pr_url}"
        return f"Error creating Pull Request: HTTP {response.status_code} - {response.text}"

# Tool 7: list_repository_files
def list_repository_files(repo: str, branch: str = "main") -> str:
    """
    Fetches the recursive list of all file paths in the repository for the specified branch.
    Returns a string containing the list of all file paths.
    """
    print(f"AGENT ACTION: Listing all files in repository branch {branch}...")
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token: headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
    with httpx.Client() as client:
        response = client.get(url, headers=headers)
        if response.status_code == 200:
            tree = response.json().get("tree", [])
            file_paths = [item.get("path") for item in tree if item.get("type") == "blob"]
            return "\n".join(file_paths)
        return f"Error listing repository files: HTTP {response.status_code} - {response.text}"


# Initialize the ADK Agent
cicd_agent = Agent(
    name="cicd_self_healing_agent",
    model=GeminiWithGroqFallback(),
    instruction="""
    You are an autonomous CI/CD self-healing agent.
    When a pipeline fails, you are given the repository name and run ID (and optionally the base_sha).
    You must automatically execute the following workflow:
    1. Call list_repository_files(repo) to fetch the entire repository file structure.
    2. Call fetch_run_jobs(repo, run_id) to read the failure logs.
    3. Analyze the logs to identify the failing test/file and cross-reference it with the repository structure to get the exact file path.
    4. Call get_file_contents(repo, failing_file_path) to get the broken source code.
    5. If the broken code imports or depends on other files (e.g. utility functions, helper components, or configurations) that are relevant to understanding/fixing the bug, call get_file_contents(repo, dependency_path) for those files as well.
    6. Generate a logical fix for the source code.
    7. Generate a new, descriptive branch name.
    8. Call create_branch(repo, base_sha, branch_name) to create a new branch.
    9. Call get_file_sha(repo, failing_file_path) to get the file's blob SHA.
    10. Call update_file(repo, failing_file_path, fixed_content, branch_name, file_sha) to commit the fix.
    11. Call create_pull_request(repo, branch_name, "main", "AI Auto-Fix: Resolved CI/CD Pipeline Failure", pr_body) to create a Pull Request. Specify in pr_body the details of the problem, the fix, and any modified files.

    Work step by step and confirm when the Pull Request is successfully created.
    """,
    tools=[fetch_run_jobs, get_file_contents, create_branch, update_file, get_file_sha, create_pull_request, list_repository_files]
)

if __name__ == "__main__":
    test_repo = "senthilkumaranT/frontend_for_self_healing_cicd"
    test_run_id = "27607186455"
    test_head_sha = "080dea81006e23606076ccf4a6f3f0293c41b87e"

    # Run the agent!
    prompt = f"The pipeline failed for repo {test_repo} on run {test_run_id}. The base commit SHA for branching is {test_head_sha}. Please fix the bug."
    print("Starting Autonomous Workflow...")

    import asyncio
    from google.adk.runners import InMemoryRunner

    async def run_test():
        runner = InMemoryRunner(agent=cicd_agent)
        try:
            res = await runner.run_debug(prompt)
            print("Workflow Complete!")
            for event in res:
                print(event)
        except Exception as e:
            print("Error executing agent:", e)
            import traceback
            traceback.print_exc()

    asyncio.run(run_test())
