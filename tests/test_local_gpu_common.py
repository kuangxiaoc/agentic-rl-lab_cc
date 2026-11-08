import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import local_gpu_common as common  # noqa: E402


class PureAlgorithmTests(unittest.TestCase):
    def test_scalar_loss_formulas(self):
        old = -1.0
        current = -0.5
        advantage = 2.0
        losses = common.scalar_losses(old, current, advantage)
        ratio = 1.6487212707001282
        self.assertAlmostEqual(losses["importance_sampling"], -ratio * advantage)
        self.assertAlmostEqual(losses["ppo"], -min(ratio * advantage, 1.2 * advantage))
        self.assertAlmostEqual(losses["cispo"], -(1.2 * current * advantage))
        with self.assertRaises(ValueError):
            common.scalar_losses(old, current, advantage, loss_fn="unknown")

    def test_group_advantages_are_mean_centered(self):
        rewards = [1.0, 0.0, 1.0, 0.0]
        advantages = common.group_advantages(rewards)
        self.assertEqual(advantages, [0.5, -0.5, 0.5, -0.5])
        self.assertAlmostEqual(sum(advantages), 0.0)
        with self.assertRaises(ValueError):
            common.group_advantages([])

    def test_valid_length_stops_after_first_eos(self):
        tokens = [10, 11, 99, 12, 13]
        self.assertEqual(common.completion_valid_length(tokens, {99}), 3)
        self.assertEqual(common.completion_valid_length(tokens, set()), len(tokens))


class OriginalCodeReuseTests(unittest.TestCase):
    def test_original_grpo_pure_logic_is_importable_without_pytrio(self):
        module = common.load_original_module(
            "grpo_03", REPO_ROOT / "experiments" / "grpo" / "03-grpo.py"
        )
        self.assertEqual(module.extract_boxed("answer \\boxed{1,000}"), "1,000")
        self.assertEqual(module.normalize_answer("1,000."), "1000")
        self.assertEqual(module.extract_gsm8k_answer("steps #### 42"), "42")
        self.assertEqual(module.grade_answer("\\boxed{42}", "42"), 1.0)
        self.assertEqual(module.grade_answer("\\boxed{41}", "42"), 0.0)

    def test_original_vision_pure_logic_is_importable_without_pytrio(self):
        module = common.load_original_module(
            "vision_train", REPO_ROOT / "experiments" / "vision-grpo" / "train.py"
        )
        self.assertEqual(module.extract_choice("final \\boxed{b}"), "B")
        self.assertIsNone(module.extract_choice("no answer"))
        self.assertIn("几何选择题", module.format_question("题", ["一", "二", "三", "四"]))


if __name__ == "__main__":
    unittest.main()
