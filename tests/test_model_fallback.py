import base64
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from agent import GeminiWithGroqFallback, create_branch, fetch_run_jobs, get_file_contents, get_file_sha
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

    def get(self, url, headers, params=None):
        self.request_headers.append(headers)
        return self.responses.pop(0)


class RecordingHttpClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def get(self, url, headers, params=None):
        self.calls.append((url, headers, params))
        return self.response

    def post(self, url, headers, json):
        self.calls.append((url, headers, json))
        return self.response


class ActionLogsClient:
    def __init__(self, log_text):
        self.log_text = log_text

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def get(self, url, headers):
        if url.endswith("/jobs"):
            return StubResponse(
                200,
                "",
                {"jobs": [{"id": 9, "name": "unit-tests", "conclusion": "failure"}]},
            )
        return StubResponse(200, self.log_text)


class RunnerStub:
    def __init__(self, session):
        self.session = session
        self.session_service = MagicMock()
        self.session_service.create_session.return_value = session
        self.run_args = None

    async def run_async(self, **kwargs):
        self.run_args = kwargs
        yield "event-1"
        yield "event-2"


class StaticLlm(BaseLlm):
    model: str = "static-test-model"

    async def generate_content_async(self, llm_request, stream=False):
        yield model_response("Runner completed without external API calls.")


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

        mock_client.assert_called_once_with(follow_redirects=True, timeout=30)
        self.assertTrue(
            all(
                headers["Authorization"] == "Bearer test-token"
                for headers in mock_client.return_value.request_headers
            )
        )
        self.assertIn("Logs for job: tests", logs)
        self.assertIn("AssertionError: expected 1, got 2", logs)


class GitHubRefTests(unittest.TestCase):
    def test_get_file_contents_uses_requested_ref_and_token(self):
        response = StubResponse(
            200,
            "",
            {"content": base64.b64encode(b"source").decode()},
        )
        client = RecordingHttpClient(response)

        with (
            patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}),
            patch("agent.httpx.Client", return_value=client),
        ):
            content = get_file_contents("owner/repo", "src/utils.py", "test1")

        self.assertEqual(content, "source")
        self.assertEqual(client.calls[0][2], {"ref": "test1"})
        self.assertEqual(
            client.calls[0][1]["Authorization"],
            "Bearer test-token",
        )

    def test_get_file_sha_uses_requested_ref(self):
        response = StubResponse(200, "", {"sha": "blob-sha"})
        client = RecordingHttpClient(response)

        with (
            patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}),
            patch("agent.httpx.Client", return_value=client),
        ):
            sha = get_file_sha("owner/repo", "src/utils.py", "failing-sha")

        self.assertEqual(sha, "blob-sha")
        self.assertEqual(client.calls[0][2], {"ref": "failing-sha"})

    def test_create_branch_requires_unique_prpatch_name(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}):
            result = create_branch(
                "owner/repo",
                "abcdef012345",
                "fix/failure",
                "123",
            )

        self.assertIn("must start with prpatch/", result)

    def test_create_branch_uses_unique_prpatch_name(self):
        client = RecordingHttpClient(StubResponse(201, "", {}))
        branch = "prpatch/fix-average-abcdef0-123"
        with (
            patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}),
            patch("agent.httpx.Client", return_value=client),
        ):
            result = create_branch("owner/repo", "abcdef012345", branch, "123")

        self.assertIn(branch, result)
        self.assertEqual(
            client.calls[0][2],
            {"ref": f"refs/heads/{branch}", "sha": "abcdef012345"},
        )


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
            "branch": "test1",
            "head_sha": "abcdef012345",
            "target_branch": "test1",
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
            "abcdef012345",
            None,
            "test1",
        )

    def test_missing_target_branch_defaults_to_main(self):
        client = TestClient(main.app)
        process_failure = AsyncMock()

        with (
            patch.object(main, "WEBHOOK_SECRET", "test-secret"),
            patch("main.fetch_and_log_github_failure", new=process_failure),
        ):
            response = client.post(
                "/webhook",
                json={"repo": "owner/repo", "run_id": "124"},
                headers={"Authorization": "Bearer test-secret"},
            )

        self.assertEqual(response.status_code, 200)
        process_failure.assert_awaited_once_with(
            "owner/repo",
            "124",
            None,
            None,
            "main",
        )


class FailureProcessingTests(unittest.IsolatedAsyncioTestCase):
    def test_render_log_excerpt_truncates_and_redacts_secrets(self):
        lines = "\n".join(f"line-{index}" for index in range(110))
        with patch.dict(os.environ, {"GITHUB_TOKEN": "do-not-log-token"}):
            excerpt = main._last_lines(f"{lines}\ndo-not-log-token", limit=100)

        self.assertIn("line-109", excerpt)
        self.assertNotIn("line-0\n", excerpt)
        self.assertNotIn("do-not-log-token", excerpt)
        self.assertIn("[REDACTED]", excerpt)

    async def test_reads_failing_commit_and_reports_pr_summary(self):
        log_text = "\n".join(f"output line-{index}" for index in range(110))
        events = [
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            text=(
                                "Root cause: src/utils.py::is_even. "
                                "Successfully created Pull Request: "
                                "https://github.com/owner/repo/pull/7"
                            )
                        )
                    ]
                )
            )
        ]
        http_client = ActionLogsClient(log_text)

        with (
            patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}),
            patch("main.httpx.Client", return_value=http_client),
            patch("main.run_agent", new=AsyncMock(return_value=events)) as run_agent,
            self.assertLogs("webhook_handler", level="INFO") as captured,
        ):
            await main.fetch_and_log_github_failure(
                "owner/repo",
                "123",
                "abcdef012345",
                target_branch="test1",
            )

        prompt = run_agent.await_args.args[1]
        self.assertIn("failing branch is test1", prompt)
        self.assertIn("at failing commit abcdef012345", prompt)
        output = "\n".join(captured.output)
        self.assertIn("output line-109", output)
        self.assertNotIn("output line-0", output)
        self.assertIn("Root cause: src/utils.py::is_even", output)
        self.assertIn("https://github.com/owner/repo/pull/7", output)

    async def test_run_agent_supports_async_session_creation(self):
        session = SimpleNamespace(id="session-id")
        runner = RunnerStub(session)
        runner.session_service.create_session = AsyncMock(return_value=session)

        with patch("google.adk.runners.InMemoryRunner", return_value=runner) as factory:
            events = await main.run_agent(MagicMock(), "fix the failure")

        self.assertEqual(events, ["event-1", "event-2"])
        factory.assert_called_once()
        self.assertEqual(factory.call_args.kwargs["app_name"], "prpatch")
        runner.session_service.create_session.assert_awaited_once_with(
            app_name="prpatch",
            user_id="ci",
        )
        self.assertEqual(runner.run_args["session_id"], "session-id")
        self.assertEqual(runner.run_args["new_message"].parts[0].text, "fix the failure")

    async def test_run_agent_supports_sync_session_creation(self):
        session = SimpleNamespace(id="session-id")
        runner = RunnerStub(session)

        with patch("google.adk.runners.InMemoryRunner", return_value=runner):
            events = await main.run_agent(MagicMock(), "fix the failure", user_id="worker")

        self.assertEqual(events, ["event-1", "event-2"])
        runner.session_service.create_session.assert_called_once_with(
            app_name="prpatch",
            user_id="worker",
        )
        self.assertEqual(runner.run_args["user_id"], "worker")

    async def test_run_agent_executes_real_runner_with_fake_model(self):
        agent = Agent(name="runner_compatibility_test", model=StaticLlm())

        events = await main.run_agent(agent, "Say the test response.")

        self.assertIn(
            "Runner completed without external API calls.",
            main._event_text(events),
        )
