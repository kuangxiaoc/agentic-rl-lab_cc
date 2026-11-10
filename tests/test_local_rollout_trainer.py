"""Dependency-free tests for multi-turn local rollout helpers."""

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "search-r1"))

import local_gpu_trainer as trainer  # noqa: E402
import local_rollout_trainer as rollout_trainer  # noqa: E402
import local_search  # noqa: E402

import importlib.util  # noqa: E402


def _load_protocol(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "experiments" / "retool" / "protocol.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


retool_protocol = _load_protocol("retool_protocol_for_test")


class MultiturnSampleTests(unittest.TestCase):
    def test_prefix_extension_keeps_only_assistant_advantages(self):
        turns = [
            rollout_trainer.TurnLike(
                prompt_tokens=[1, 2],
                completion_tokens=[3, 4],
                logprobs=[-0.1, -0.2],
            ),
            rollout_trainer.TurnLike(
                prompt_tokens=[1, 2, 3, 4, 5, 6],
                completion_tokens=[7],
                logprobs=[-0.3],
            ),
        ]
        sample = rollout_trainer.build_multiturn_sample(
            turns, advantage=0.5, reward=1.0, text="answer"
        )
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.input_ids, [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(sample.completion_ids, [3, 4, 5, 6, 7])
        self.assertEqual(sample.old_logprobs, [-0.1, -0.2, 0.0, 0.0, -0.3])
        self.assertEqual(sample.advantages, [0.5, 0.5, 0.0, 0.0, 0.5])
        self.assertEqual(sample.reward, 1.0)
        self.assertEqual(sample.text, "answer")

    def test_non_prefix_prompt_is_rejected(self):
        turns = [
            rollout_trainer.TurnLike([1, 2], [3], [-0.1]),
            rollout_trainer.TurnLike([9, 9, 9], [4], [-0.2]),
        ]
        with self.assertRaises(ValueError):
            rollout_trainer.build_multiturn_sample(turns, 1.0)

    def test_mismatched_logprobs_are_rejected(self):
        turns = [rollout_trainer.TurnLike([1], [2, 3], [-0.1])]
        with self.assertRaises(ValueError):
            rollout_trainer.build_multiturn_sample(turns, 1.0)

    def test_empty_turns_return_none(self):
        self.assertIsNone(rollout_trainer.build_multiturn_sample([], 1.0))

    def test_forced_prefix_tokens_receive_zero_advantage(self):
        turns = [
            rollout_trainer.TurnLike(
                prompt_tokens=[1, 2],
                completion_tokens=[3, 4, 5],
                logprobs=[0.0, -0.1, -0.2],
                forced_prefix_tokens=1,
            )
        ]
        sample = rollout_trainer.build_multiturn_sample(turns, advantage=2.0)
        assert sample is not None
        self.assertEqual(sample.advantages, [0.0, 2.0, 2.0])


class LocalTrioCompatibilityTests(unittest.TestCase):
    def test_token_prompt_and_sampling_params_shapes(self):
        trio = trainer.local_trio_compat()
        prompt = trio.ModelInput.from_ints([1, 2, 3])
        self.assertEqual(prompt.tokens, [1, 2, 3])
        params = trio.SamplingParams(max_tokens=16, seed=7, stop=["x"])
        self.assertEqual(params.max_tokens, 16)
        self.assertEqual(params.seed, 7)
        self.assertEqual(params.stop, ["x"])


class ToolProtocolTests(unittest.TestCase):
    def test_qwen3_instruct_json_tool_call_is_parsed(self):
        parsed = retool_protocol.parse_assistant(
            '<tool_call>\n'
            '{"name": "code_interpreter", '
            '"arguments": {"code": "print(12 * 7)"}}\n'
            "</tool_call>"
        )
        self.assertEqual(parsed.kind, "tool")
        self.assertEqual(parsed.code, "print(12 * 7)")

    def test_json_tool_call_allows_braces_inside_generated_code(self):
        parsed = retool_protocol.parse_assistant(
            '<tool_call>\n'
            '{"name": "code_interpreter", "arguments": '
            '{"code": "values = {1: 2}\\nprint(values)"}}\n'
            "</tool_call>"
        )
        self.assertEqual(parsed.kind, "tool")
        self.assertEqual(parsed.code, "values = {1: 2}\nprint(values)")

    def test_malformed_json_call_is_invalid_not_a_tool_call(self):
        parsed = retool_protocol.parse_assistant(
            '<tool_call>{"name": "code_interpreter"</tool_call>'
        )
        self.assertEqual(parsed.kind, "invalid")
        self.assertIsNone(parsed.code)

    def test_markdown_fence_is_removed_from_tool_code(self):
        parsed = retool_protocol.parse_assistant(
            '<tool_call>\n'
            '{"name": "code_interpreter", "arguments": '
            '{"code": "```python\\nprint(12 * 7)\\n```"}}\n'
            "</tool_call>"
        )
        self.assertEqual(parsed.kind, "tool")
        self.assertEqual(parsed.code, "print(12 * 7)")

    def test_backticks_inside_python_are_not_aggressively_stripped(self):
        code = 'value = "`not a fence`"\nprint(value)'
        self.assertEqual(retool_protocol.strip_markdown_fence(code), code)

    def test_tool_scaffold_leaves_only_the_python_body_to_generate(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(text.encode())

            def decode(self, tokens, skip_special_tokens=True):
                return bytes(tokens).decode()

        prefix = bytes(retool_protocol.tool_call_prefix(Tokenizer())).decode()
        self.assertIn('"name": "code_interpreter"', prefix)
        self.assertTrue(prefix.endswith('"code": "'))
        self.assertTrue(retool_protocol.tool_call_suffix().endswith("</tool_call>"))
        complete = retool_protocol.tool_call_prefix(Tokenizer()) + list(
            b'print(1)"}}\n</tool_call>'
        )
        self.assertEqual(
            retool_protocol.complete_tool_call_text(Tokenizer(), complete).count(
                "</tool_call>"
            ),
            1,
        )


class OfflineSearchTests(unittest.TestCase):
    class Example:
        def __init__(self, question, answers):
            self.question = question
            self.answers = answers

    def test_lexical_search_is_explicitly_oracle_like(self):
        client = local_search.OfflineEvidenceSearch(
            [
                self.Example("who wrote hamlet", ["Shakespeare"]),
                self.Example("what is the capital of France", ["Paris"]),
            ],
            max_results=2,
        )
        result = client.search("capital France")
        self.assertTrue(result.ok)
        self.assertEqual(result.items[0].title, "what is the capital of France")
        self.assertIn("Reference answer: Paris", result.items[0].content)
        self.assertIn("offline oracle evidence", result.items[0].content)
        self.assertEqual(client.stats.successes, 1)

    def test_empty_query_is_an_error_without_fake_results(self):
        client = local_search.OfflineEvidenceSearch(
            [self.Example("who wrote hamlet", ["Shakespeare"])]
        )
        result = client.search("the and")
        self.assertFalse(result.ok)
        self.assertEqual(result.items, [])
        self.assertEqual(client.stats.errors, 1)


if __name__ == "__main__":
    unittest.main()
