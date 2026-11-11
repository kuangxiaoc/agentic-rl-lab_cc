"""Dependency-free tests for the AgentOPSD tool protocol."""

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "experiments" / "agentopsd" / "protocol.py"


def load_protocol():
    spec = importlib.util.spec_from_file_location("agentopsd_protocol_for_test", PROTOCOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


agentopsd_protocol = load_protocol()


class AgentOPSDProtocolTests(unittest.TestCase):
    def test_generation_stops_at_tool_call_close(self):
        tokenizer = type("Tokenizer", (), {"eos_token": "<|im_end|>"})()
        self.assertEqual(
            agentopsd_protocol.stop_sequences(tokenizer),
            ["<|im_end|>", "</tool_call>"],
        )

    def test_generation_without_eos_still_stops_at_tool_call_close(self):
        tokenizer = type("Tokenizer", (), {"eos_token": None})()
        self.assertEqual(agentopsd_protocol.stop_sequences(tokenizer), ["</tool_call>"])


if __name__ == "__main__":
    unittest.main()
