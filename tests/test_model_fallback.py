import os
import unittest
import base64
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from agent import (
    GeminiWithGroqFallback,
    fetch_run_jobs,
    get_file_contents,
    get_file_sha,
)
import main


class StubModel:
    def __init__(self, responses=(), error=None):
        self.responses = responses
        self.error = error

    async def generate_content_async(self, llm_request, stream=False):
        for response in self.responses:
            yield response
        if self.error:
            raise self.error


def model_response(text):
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)])
    )


class StubResponse:
    def __init__(self, status_code, text, data=None):
        self.status_code = status_code
        self.text = text
        self.data = data or {}

    def json(self):
        return self.data


class StubHttpClient:
    def __init__(self, **kwargs):
        self.follow_redirects = kwargs.get("follow_redirects", False)
        self.responses = [
            StubResponse(
                200,
                '{"jobs":[{"id":123,"name":"tests","conclusion":"failure"}]}',
                {"jobs": [{"id": 123, "name": "tests", "conclusion": "failure"}]},
            ),
            StubResponse(200, "AssertionError: expected 1, got 2"),
        ]
        self.request_headers = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def get(self, url, headers):
        self.request_headers.append(headers)
        return self.responses.pop(0)


class RecordingContentClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def get(self, url, headers, params=None):
        self.calls.append((url, params))
        return self.response


class GeminiWithGroqFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_groq_when_gemini_fails_before_responding(self):
        gemini = StubModel(error=RuntimeError("Gemini unavailable"))
        groq = StubModel(responses=[model_response("Groq response")])

        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}),
            patch("agent.Gemini", return_value=gemini),
            patch("google.adk.models.lite_llm.LiteLlm", return_value=groq),
        ):
            responses = [
                response
                async for response in GeminiWithGroqFallback()
                .generate_content_async(None)
            ]

        self.assertEqual(responses[0].content.parts[0].text, "Groq response")

    async def test_does_not_retry_after_gemini_has_started_responding(self):
        gemini = StubModel(
            responses=[model_response("Partial response")],
            error=RuntimeError("Gemini stream failed"),
        )

        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}),
            patch("agent.Gemini", return_value=gemini),
            patch("google.adk.models.lite_llm.LiteLlm") as groq_constructor,
        ):
            responses = []
            with self.assertRaisesRegex(RuntimeError, "Gemini stream failed"):
                async for response in GeminiWithGroqFallback().generate_content_async(None):
                    responses.append(response)

        self.assertEqual(len(responses), 1)
        groq_constructor.assert_not_called()


class FetchRunJobsTests(unittest.TestCase):
    def test_returns_failed_job_logs(self):
        client_factory = patch(
            "agent.httpx.Client", return_value=StubHttpClient()
        )
        with (
            patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}),
            client_factory as mock_client,
        ):
            logs = fetch_run_jobs("owner/repo", "123")

        mock_client.assert_called_once_with(follow_redirects=True)
        self.assertTrue(
            all(
                headers["Authorization"] == "Bearer test-token"
                for headers in mock_client.return_value.request_headers
            )
        )
        self.assertIn("Logs for job: tests", logs)
        self.assertIn("AssertionError: expected 1, got 2", logs)


class RepositoryRefTests(unittest.TestCase):
    def test_get_file_contents_reads_requested_commit(self):
        client = RecordingContentClient(
            StubResponse(
                200,
                "",
                {"content": base64.b64encode(b"source at commit").decode()},
            )
        )
        with patch("agent.httpx.Client", return_value=client):
            content = get_file_contents("owner/repo", "src/file.py", "commit-sha")

        self.assertEqual(content, "source at commit")
        self.assertEqual(client.calls[0][1], {"ref": "commit-sha"})

    def test_get_file_sha_reads_requested_commit(self):
        client = RecordingContentClient(StubResponse(200, "", {"sha": "blob-sha"}))
        with patch("agent.httpx.Client", return_value=client):
            sha = get_file_sha("owner/repo", "src/file.py", "commit-sha")

        self.assertEqual(sha, "blob-sha")
        self.assertEqual(client.calls[0][1], {"ref": "commit-sha"})


class WebhookAuthTests(unittest.TestCase):
    def test_health_endpoint(self):
        response = TestClient(main.app).get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_requires_configured_bearer_secret(self):
        client = TestClient(main.app)
        payload = {
            "repo": "owner/repo",
            "run_id": "123",
            "branch": "feature/source",
            "head_sha": "commit-sha",
            "target_branch": "feature/source",
        }
        process_failure = AsyncMock()

        with (
            patch.object(main, "WEBHOOK_SECRET", "test-secret"),
            patch(
                "main.fetch_and_log_github_failure",
                new=process_failure,
            ),
        ):
            unauthorized = client.post(
                "/webhook",
                json=payload,
                headers={"Authorization": "Bearer wrong-secret"},
            )
            accepted = client.post(
                "/webhook",
                json=payload,
                headers={"Authorization": "Bearer test-secret"},
            )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(accepted.status_code, 200)
        process_failure.assert_awaited_once_with(
            "owner/repo",
            "123",
            "commit-sha",
            None,
            "feature/source",
        )


class RepairPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_repair_prompt_uses_requested_target_branch(self):
        runner = MagicMock()
        runner.run_debug = AsyncMock(return_value=[])

        with patch("google.adk.runners.InMemoryRunner", return_value=runner):
            await main.fetch_and_log_github_failure(
                "owner/repo",
                "123",
                "commit-sha",
                "test log",
                "feature/source",
            )

        prompt = runner.run_debug.await_args.args[0]
        self.assertIn("Create the repair pull request against branch feature/source", prompt)
        self.assertIn("Use this exact SHA as the ref", prompt)
