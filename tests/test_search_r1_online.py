"""Dependency-free tests for the Search-R1 Tavily backend."""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_PATH = REPO_ROOT / "experiments" / "search-r1" / "search.py"


class FakeHTTPStatusError(Exception):
    def __init__(self, response):
        super().__init__(f"HTTP {response.status_code}")
        self.response = response


class FakeTimeoutError(Exception):
    pass


class FakeHTTPError(Exception):
    pass


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise FakeHTTPStatusError(self)

    def json(self):
        return self._payload


class FakeHTTPX(types.ModuleType):
    HTTPStatusError = FakeHTTPStatusError
    TimeoutException = FakeTimeoutError
    HTTPError = FakeHTTPError

    def __init__(self, responses):
        super().__init__("httpx")
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError("unexpected HTTP request")
        return self.responses.pop(0)


def load_search_module(fake_httpx):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    with patch.dict(
        sys.modules,
        {"httpx": fake_httpx, "dotenv": fake_dotenv},
    ):
        spec = importlib.util.spec_from_file_location(
            "search_r1_online_test", SEARCH_PATH
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class TavilySearchClientTests(unittest.TestCase):
    def test_real_online_shape_parses_results_and_never_reprs_key(self):
        fake_httpx = FakeHTTPX(
            [
                FakeResponse(
                    {
                        "results": [
                            {
                                "title": "France",
                                "url": "https://example.com/france",
                                "content": "Paris is the capital of France.",
                            },
                            {
                                "title": "Paris",
                                "url": "https://example.com/paris",
                                "content": "Paris is a European capital city.",
                            },
                        ]
                    }
                )
            ]
        )
        module = load_search_module(fake_httpx)
        client = module.TavilySearchClient(
            api_key="test-key-not-real",
            count=2,
            timeout=7.0,
        )

        result = client.search(" capital of France ")

        self.assertTrue(result.ok)
        self.assertEqual([item.title for item in result.items], ["France", "Paris"])
        self.assertEqual(result.items[0].source, "Tavily")
        self.assertEqual(result.items[0].url, "https://example.com/france")
        self.assertNotIn("test-key-not-real", repr(client))
        self.assertEqual(client.stats.requests, 1)
        self.assertEqual(client.stats.api_requests, 1)
        self.assertEqual(client.stats.successes, 1)
        self.assertEqual(client.stats.billable_requests, 1)
        self.assertEqual(client.stats.result_count, 2)
        url, request = fake_httpx.calls[0]
        self.assertEqual(url, module.TAVILY_SEARCH_ENDPOINT)
        self.assertFalse(request["trust_env"])
        self.assertEqual(request["json"]["query"], "capital of France")
        self.assertEqual(request["json"]["max_results"], 2)
        self.assertEqual(request["json"]["search_depth"], "basic")

    def test_rate_limit_retries_and_counts_actual_api_attempts(self):
        fake_httpx = FakeHTTPX(
            [
                FakeResponse({"detail": "rate limited"}, status_code=429),
                FakeResponse({"results": []}),
            ]
        )
        module = load_search_module(fake_httpx)
        client = module.TavilySearchClient(
            api_key="test-key-not-real",
            max_retries=1,
            retry_delay=0.0,
        )

        result = client.search("query")

        self.assertTrue(result.ok)
        self.assertEqual(len(fake_httpx.calls), 2)
        self.assertEqual(client.stats.requests, 1)
        self.assertEqual(client.stats.api_requests, 2)
        self.assertEqual(client.stats.rate_limits, 1)
        self.assertEqual(client.stats.successes, 1)

    def test_invalid_query_does_not_call_tavily(self):
        fake_httpx = FakeHTTPX([])
        module = load_search_module(fake_httpx)
        client = module.TavilySearchClient(api_key="test-key-not-real")

        result = client.search("   ")

        self.assertFalse(result.ok)
        self.assertEqual(result.status, 400)
        self.assertEqual(fake_httpx.calls, [])
        self.assertEqual(client.stats.requests, 1)
        self.assertEqual(client.stats.api_requests, 0)

    def test_missing_api_key_fails_before_any_request(self):
        fake_httpx = FakeHTTPX([])
        module = load_search_module(fake_httpx)
        with patch.dict(os.environ, {"TAVILY_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "TAVILY_API_KEY"):
                module.TavilySearchClient()
        self.assertEqual(fake_httpx.calls, [])


if __name__ == "__main__":
    unittest.main()
