import base64
import json
import httpx
import pytest
from fastapi.testclient import TestClient
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

import main
import pipeline
from agent import RepairContext, build_agent
from github_api import GitHub
from log_utils import extract_relevant
from main import app


def test_log_trim_keeps_error_and_is_small():
    lines = [f"2026-10-17T12:00:00.000Z info: step {i} executed without issues" for i in range(250)]
    lines.append("\x1b[31mAssertionError: expected 2 to equal 3\x1b[0m")
    lines.extend([f"2026-10-17T12:00:01.000Z debug: trailing log line {i} with additional system noise" for i in range(250)])
    raw_log = "\n".join(lines)

    excerpt = extract_relevant(raw_log)
    assert "AssertionError" in excerpt
    assert "\x1b" not in excerpt
    assert len(excerpt) <= 12100


def test_auth(monkeypatch):
    client = TestClient(app)
    payload = {"repo": "owner/repo", "run_id": "123", "head_sha": "abc1234"}

    # 1. With WEBHOOK_SECRET unset
    monkeypatch.delenv("WEBHOOK_SECRET", raising=False)
    resp = client.post("/webhook", json=payload, headers={"Authorization": "Bearer None"})
    assert resp.status_code == 503

    # 2. With WEBHOOK_SECRET set to s3cret
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cret")

    async def fake_run_repair(*args, **kwargs):
        return {"status": "ok"}

    monkeypatch.setattr("main.run_repair", fake_run_repair)

    # No header -> 401
    resp = client.post("/webhook", json=payload)
    assert resp.status_code == 401

    # Wrong header -> 401
    resp = client.post("/webhook", json=payload, headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401

    # Correct header -> 202
    resp = client.post("/webhook", json=payload, headers={"Authorization": "Bearer s3cret"})
    assert resp.status_code == 202
    assert resp.json()["status"] == "accepted"


def test_loop_guard():
    assert pipeline.should_skip("o/r", "prpatch/fix-1-abc", "x")


class FakeLlm(BaseLlm):
    model: str = "fake"
    responses: list = []
    idx: int = 0

    async def generate_content_async(self, llm_request, stream: bool = False):
        resp = self.responses[self.idx]
        self.idx += 1
        yield resp


@pytest.mark.asyncio
async def test_end_to_end(monkeypatch):
    pipeline._seen.clear()

    put_calls = []
    pr_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        method = request.method
        if "/actions/runs/9/jobs" in url:
            return httpx.Response(
                200,
                json={"jobs": [{"id": 101, "name": "test-job", "conclusion": "failure"}, {"id": 102, "name": "lint-job", "conclusion": "success"}]},
            )
        elif "/actions/jobs/101/logs" in url:
            return httpx.Response(200, text="error: AssertionError in test_add")
        elif "/git/trees/abc123" in url:
            return httpx.Response(200, json={"tree": [{"path": "calc.py", "type": "blob"}]})
        elif "/contents/calc.py" in url and method == "GET":
            return httpx.Response(
                200,
                json={
                    "content": base64.b64encode(b"def add(a, b):\n    return a - b\n").decode(),
                    "sha": "orig_sha",
                    "encoding": "base64",
                },
            )
        elif "/git/refs" in url and method == "POST":
            return httpx.Response(201, json={"ref": "refs/heads/prpatch/fix-9-abc123"})
        elif "/contents/calc.py" in url and method == "PUT":
            put_calls.append(json.loads(request.content.decode()))
            return httpx.Response(200, json={"content": {}})
        elif "/pulls" in url and method == "POST":
            pr_calls.append(json.loads(request.content.decode()))
            return httpx.Response(201, json={"html_url": "https://github.com/o/r/pull/42"})
        return httpx.Response(404, text=f"not found: {method} {url}")

    transport = httpx.MockTransport(handler)
    gh = GitHub("t", transport=transport)

    fixed_content = "def add(a, b):\n    return a + b\n"
    fake_llm = FakeLlm(
        responses=[
            LlmResponse(content=types.Content(parts=[types.Part.from_function_call(name="list_repository_files", args={})])),
            LlmResponse(content=types.Content(parts=[types.Part.from_function_call(name="read_file", args={"path": "calc.py"})])),
            LlmResponse(
                content=types.Content(
                    parts=[
                        types.Part.from_function_call(
                            name="submit_fix",
                            args={"file_path": "calc.py", "new_content": fixed_content, "explanation": "fixed addition"},
                        )
                    ]
                )
            ),
            LlmResponse(content=types.Content(parts=[types.Part.from_text(text="Fixed.")])),
        ]
    )

    orig_build_agent = pipeline.build_agent

    def fake_build_agent(ctx):
        ag = orig_build_agent(ctx)
        ag.model = fake_llm
        return ag

    monkeypatch.setattr(pipeline, "build_agent", fake_build_agent)

    res = await pipeline.run_repair("o/r", "9", "abc123", "feature", gh=gh)

    assert len(put_calls) == 1
    assert put_calls[0]["branch"] == "prpatch/fix-9-abc123"
    assert base64.b64decode(put_calls[0]["content"]).decode("utf-8") == fixed_content

    assert len(pr_calls) == 1
    assert pr_calls[0]["base"] == "feature"

    assert res == {"status": "pr_opened", "url": "https://github.com/o/r/pull/42"}


@pytest.mark.asyncio
async def test_no_fix_path(monkeypatch):
    pipeline._seen.clear()

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/actions/runs/9/jobs" in url:
            return httpx.Response(200, json={"jobs": [{"id": 101, "name": "test-job", "conclusion": "failure"}]})
        elif "/actions/jobs/101/logs" in url:
            return httpx.Response(200, text="error: unknown error occurred")
        elif "/repos/o/r" in url:
            return httpx.Response(200, json={"default_branch": "main"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    gh = GitHub("t", transport=transport)

    fake_llm = FakeLlm(
        responses=[
            LlmResponse(content=types.Content(parts=[types.Part.from_text(text="Could not determine fix.")])),
        ]
    )

    orig_build_agent = pipeline.build_agent

    def fake_build_agent(ctx):
        ag = orig_build_agent(ctx)
        ag.model = fake_llm
        return ag

    monkeypatch.setattr(pipeline, "build_agent", fake_build_agent)

    res = await pipeline.run_repair("o/r", "9", "abc123", "feature", gh=gh)
    assert res["status"] == "no_fix"


@pytest.mark.asyncio
async def test_submit_fix_guards():
    def handler(request: httpx.Request) -> httpx.Response:
        if "/contents/calc.py" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "content": base64.b64encode(b"def add(a, b):\n    return a - b\n").decode(),
                    "sha": "calcsha",
                    "encoding": "base64",
                },
            )
        return httpx.Response(404)

    gh = GitHub("t", transport=httpx.MockTransport(handler))
    ctx = RepairContext(repo="o/r", ref="abc123", gh=gh)
    tools = {t.__name__: t for t in build_agent(ctx).tools}
    read_file = tools["read_file"]
    submit_fix = tools["submit_fix"]

    # 1. Blocked paths
    r1 = await read_file(".github/workflows/ci.yml")
    assert "not allowed" in r1

    r2 = await read_file("../x")
    assert "not allowed" in r2

    # 2. submit_fix before reading
    s1 = await submit_fix("calc.py", "def add(): pass", "expl")
    assert "read_file" in s1

    # Read the file
    await read_file("calc.py")

    # 3. Invalid Python
    s2 = await submit_fix("calc.py", "def broken(:", "expl")
    assert "not valid" in s2

    # 4. Identical content
    s3 = await submit_fix("calc.py", "def add(a, b):\n    return a - b\n", "expl")
    assert "identical" in s3

    # 5. Real fix
    s4 = await submit_fix("calc.py", "def add(a, b):\n    return a + b\n", "fix addition")
    assert s4.startswith("OK")

    # 6. Second submit
    s5 = await submit_fix("calc.py", "def add(a, b):\n    return a + b\n", "fix addition")
    assert "already" in s5
