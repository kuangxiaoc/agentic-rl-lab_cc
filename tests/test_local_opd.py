"""Dependency-free tests for local OPD and Jev data helpers."""

import sys
import unittest
import importlib.util
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "pytrio-jev"))

import local_opd  # noqa: E402
import local_gpu_data as jev_data  # noqa: E402


def load_medical_module():
    path = REPO_ROOT / "experiments" / "medical-opd" / "local_gpu.py"
    spec = importlib.util.spec_from_file_location("medical_local_gpu", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    def __init__(self, vocab_size=10, mapping=None):
        self.vocab_size = vocab_size
        self.mapping = mapping or {"a": [1, 2], "b": [1, 3]}

    def encode(self, text, add_special_tokens=False):
        return list(self.mapping.get(text, [len(text)]))


class OPDMathTests(unittest.TestCase):
    def test_reverse_kl_advantage_is_negative_beta_times_gap(self):
        self.assertEqual(
            local_opd.reverse_kl_advantages(
                [-1.0, -2.0], [-2.0, -1.0], kl_penalty_coef=0.5
            ),
            [-0.5, 0.5],
        )

    def test_reverse_kl_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            local_opd.reverse_kl_advantages([-1.0], [-1.0, -2.0])

    def test_choice_parser_handles_think_and_answer_prefix(self):
        self.assertEqual(local_opd.parse_choice("答案是 B"), "B")
        self.assertEqual(local_opd.parse_choice("<think>x</think>\n最终答案：C"), "C")
        self.assertEqual(local_opd.parse_choice("D"), "D")

    def test_choice_prompt_has_four_options_without_answer(self):
        prompt = local_opd.build_choice_prompt(
            "题目", {"A": "一", "B": "二", "C": "三", "D": "四"}
        )
        self.assertIn("A. 一", prompt)
        self.assertIn("D. 四", prompt)
        self.assertNotIn("答案_idx", prompt)


class TokenizerCompatibilityTests(unittest.TestCase):
    def test_identical_fingerprints_pass(self):
        local_opd.assert_compatible_tokenizers(
            FakeTokenizer(),
            FakeTokenizer(),
        )

    def test_vocab_or_token_ids_differ_rejected(self):
        probe = "Qwen3 OPD tokenizer check: <|im_start|>assistant\nA B C D 1 2 3 医疗"
        with self.assertRaises(ValueError):
            local_opd.assert_compatible_tokenizers(
                FakeTokenizer(vocab_size=10),
                FakeTokenizer(vocab_size=11),
            )
        with self.assertRaises(ValueError):
            local_opd.assert_compatible_tokenizers(
                FakeTokenizer(mapping={probe: [1]}),
                FakeTokenizer(mapping={probe: [2]}),
            )

    def test_opd_update_uses_distinct_teacher_prompt(self):
        """OPSD needs the teacher to score the same completion under another prompt."""
        seen_prompts = []

        def fake_teacher(model, prompt_ids, completion_ids, **kwargs):
            seen_prompts.append(list(prompt_ids))
            return [-0.5] * len(completion_ids)

        fake_torch = mock.Mock()
        fake_torch.tensor.return_value = object()
        fake_torch.stack.return_value.mean.return_value.item.return_value = 0.0
        fake_student = mock.Mock()
        fake_student.parameters.return_value = []
        fake_loss = mock.Mock()
        fake_loss.detach.return_value = fake_loss
        with (
            mock.patch.object(local_opd, "completion_teacher_logprobs", fake_teacher),
            mock.patch.object(
                local_opd.trainer,
                "generate_group",
                return_value=([[5, 6]], [[-1.0, -1.0]], ["ok"], 2),
            ),
            mock.patch.object(local_opd.trainer, "sequence_policy_loss", return_value=fake_loss),
        ):
            local_opd.run_opd_update_with_prompts(
                student_model=fake_student,
                teacher_model=None,
                tokenizer=None,
                optimizer=mock.Mock(),
                student_prompt_ids_list=[[1, 2]],
                teacher_prompt_ids_list=[[9, 8, 7]],
                group_size=1,
                max_new_tokens=1,
                temperature=1.0,
                top_p=1.0,
                kl_penalty_coef=1.0,
                loss_fn="importance_sampling",
                torch_module=fake_torch,
                device=None,
                seed=0,
            )
        self.assertEqual(seen_prompts, [[9, 8, 7]])


class JevDataTests(unittest.TestCase):
    def record(self):
        return {
            "id": "demo",
            "state": {"message": "refund"},
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "Choose a team.",
                    "criteria": {"billing": "Payments", "technical": "Tech"},
                },
                "urgent": {
                    "type": "noul",
                    "instructions": "Is it urgent?",
                },
            },
            "targets": {
                "team": {"label": "technical"},
                "urgent": {"label": "true"},
            },
        }

    def test_targets_are_symbols_not_raw_labels(self):
        compiled = jev_data.compile_record(self.record())
        self.assertEqual(compiled["targets"], {"team": "B", "urgent": "B"})
        self.assertEqual(
            compiled["options"]["urgent"],
            [
                ("A", "no", jev_data.NOUL_NO),
                ("B", "yes", jev_data.NOUL_YES),
            ],
        )

    def test_skeleton_uses_single_marker_per_field(self):
        compiled = jev_data.compile_record(self.record())
        assistant = compiled["messages"][-1]["content"]
        self.assertEqual(assistant.count(jev_data.MARKER), 2)
        self.assertIn('"team": "decision"', assistant)

    def test_chat_text_has_minimal_qwen_boundaries(self):
        compiled = jev_data.compile_record(self.record())
        text = jev_data.build_chat_text(
            compiled["messages"][:-1],
            add_generation_prompt=True,
        )
        self.assertTrue(text.startswith("<|im_start|>system\n"))
        self.assertTrue(text.endswith("<|im_start|>assistant\n"))


class MedicalDataTests(unittest.TestCase):
    def test_completion_uses_think_tags_like_original_sft(self):
        module = load_medical_module()
        self.assertEqual(
            module.build_medical_completion("reason", "answer"),
            "<think>\nreason\n</think>\n\nanswer",
        )
        self.assertEqual(
            module.build_medical_completion("", "answer"),
            "answer",
        )


if __name__ == "__main__":
    unittest.main()
