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
            "grpo_03",
            REPO_ROOT / "experiments" / "grpo" / "03-grpo.py",
            required=("FEWSHOT_PREFIX", "QUESTION_SUFFIX", "extract_gsm8k_answer", "grade_answer"),
        )
        self.assertEqual(module.extract_boxed("answer \\boxed{1,000}"), "1,000")
        self.assertEqual(module.normalize_answer("1,000."), "1000")
        self.assertEqual(module.extract_gsm8k_answer("steps #### 42"), "42")
        self.assertEqual(module.grade_answer("\\boxed{42}", "42"), 1.0)
        self.assertEqual(module.grade_answer("\\boxed{41}", "42"), 0.0)

    def test_constants_built_from_other_constants_are_kept(self):
        """FEWSHOT_PREFIX is a list whose text adds QUESTION_SUFFIX, not a bare literal."""
        module = common.load_original_module(
            "grpo_03", REPO_ROOT / "experiments" / "grpo" / "03-grpo.py"
        )
        self.assertEqual(len(module.FEWSHOT_PREFIX), 2)
        self.assertEqual({msg["role"] for msg in module.FEWSHOT_PREFIX}, {"user", "assistant"})
        self.assertTrue(module.FEWSHOT_PREFIX[0]["content"].endswith(module.QUESTION_SUFFIX))
        self.assertIn("\\boxed{", module.FEWSHOT_PREFIX[1]["content"])

    def test_compiled_regex_constant_comes_from_the_original_file(self):
        module = common.load_original_module(
            "vision_train",
            REPO_ROOT / "experiments" / "vision-grpo" / "train.py",
            required=("BOXED_CHOICE_PATTERN", "extract_choice"),
        )
        self.assertEqual(module.BOXED_CHOICE_PATTERN.pattern, r"\\boxed\s*\{\s*([A-D])\s*\}")
        self.assertEqual(module.extract_choice("\\boxed{d}"), "D")

    def test_required_attributes_fail_loudly(self):
        with self.assertRaises(AttributeError) as context:
            common.load_original_module(
                "grpo_03",
                REPO_ROOT / "experiments" / "grpo" / "03-grpo.py",
                required=("A_CONSTANT_that_was_dropped",),
            )
        self.assertIn("did not expose", str(context.exception))

    def test_original_vision_pure_logic_is_importable_without_pytrio(self):
        module = common.load_original_module(
            "vision_train",
            REPO_ROOT / "experiments" / "vision-grpo" / "train.py",
            required=("CHOICE_LETTERS", "format_question", "extract_choice"),
        )
        self.assertEqual(module.extract_choice("final \\boxed{b}"), "B")
        self.assertIsNone(module.extract_choice("no answer"))
        self.assertIn("几何选择题", module.format_question("题", ["一", "二", "三", "四"]))


class DAPOAdvantageTests(unittest.TestCase):
    def test_normalized_advantage_uses_sample_standard_deviation(self):
        # rewards [1, 0, 1, 0]: mean 0.5, sample std sqrt(1/3) ~ 0.5774
        advantages = common.normalized_group_advantages([1, 0, 1, 0], epsilon=0.0)
        std = (1 / 3) ** 0.5
        self.assertAlmostEqual(advantages[0], 0.5 / std)
        self.assertAlmostEqual(advantages[1], -0.5 / std)
        self.assertAlmostEqual(sum(advantages), 0.0)

    def test_epsilon_keeps_a_degenerate_group_from_dividing_by_zero(self):
        self.assertEqual(common.normalized_group_advantages([1, 1, 1]), [0.0, 0.0, 0.0])
        self.assertEqual(common.normalized_group_advantages([1]), [0.0])
        with self.assertRaises(ValueError):
            common.normalized_group_advantages([])

    def test_dynamic_sampling_keeps_only_mixed_groups(self):
        self.assertTrue(common.is_effective_group([1, 0, 1, 0], 4))
        self.assertTrue(common.is_effective_group([1, 0, 0, 0], 4))
        self.assertFalse(common.is_effective_group([1, 1, 1, 1], 4))
        self.assertFalse(common.is_effective_group([0, 0, 0, 0], 4))


class DAPOLossTests(unittest.TestCase):
    def test_clip_higher_allows_more_positive_growth(self):
        # ratio 1.5 with advantage +1: PPO's 1.2 cap would clip, DAPO's 1.28 also
        # clips, so use 1.25 where the two presets disagree.
        self.assertEqual(common.clipped_objectives([1.25], [1.0], clip_high=1.2), [1.2])
        self.assertEqual(common.clipped_objectives([1.25], [1.0], clip_high=1.28), [1.25])

    def test_negative_advantage_is_bounded_by_clip_low(self):
        # ratio 0.5, advantage -1: objective = min(-0.5, -0.8) = -0.8
        self.assertEqual(common.clipped_objectives([0.5], [-1.0]), [-0.8])

    def test_token_reduction_weights_long_answers_more(self):
        sequences = [([0.5] * 4, [1.0] * 4), ([1.0], [1.0])]
        self.assertAlmostEqual(common.dapo_loss(sequences), -0.6)
        self.assertAlmostEqual(common.dapo_loss(sequences, reduction="sample"), -0.75)

    def test_loss_rejects_mismatched_and_empty_inputs(self):
        with self.assertRaises(ValueError):
            common.clipped_objectives([1.0, 1.0], [1.0])
        with self.assertRaises(ValueError):
            common.clipped_objectives([], [])
        with self.assertRaises(ValueError):
            common.dapo_loss([], reduction="token")
        with self.assertRaises(ValueError):
            common.dapo_loss([([1.0], [1.0])], reduction="bogus")


class DAPOOriginalRewardTests(unittest.TestCase):
    """Reuse experiments/dapo/reward.py without installing math_verify."""

    def load(self):
        def parse(value):
            return value.strip("$")

        def verify(left, right):
            return left == right

        return common.load_original_module(
            "dapo_reward",
            REPO_ROOT / "experiments" / "dapo" / "reward.py",
            required=(
                "extract_last_boxed",
                "answers_equivalent",
                "soft_overlong_penalty",
                "score_answer",
                "RewardResult",
            ),
            namespace={"parse": parse, "verify": verify},
        )

    def test_extract_last_boxed_handles_nested_braces(self):
        module = self.load()
        self.assertEqual(module.extract_last_boxed("first \\boxed{1} then \\boxed{2}"), "2")
        self.assertEqual(module.extract_last_boxed("\\boxed{\\frac{1}{2}}"), "\\frac{1}{2}")
        self.assertIsNone(module.extract_last_boxed("no answer"))

    def test_soft_overlong_penalty_is_piecewise_linear(self):
        module = self.load()
        self.assertEqual(module.soft_overlong_penalty(700, max_tokens=1000, overlong_cache=200), 0.0)
        self.assertEqual(module.soft_overlong_penalty(800, max_tokens=1000, overlong_cache=200), 0.0)
        self.assertAlmostEqual(
            module.soft_overlong_penalty(900, max_tokens=1000, overlong_cache=200), -0.5
        )
        self.assertEqual(module.soft_overlong_penalty(1000, max_tokens=1000, overlong_cache=200), -1.0)
        self.assertEqual(module.soft_overlong_penalty(1200, max_tokens=1000, overlong_cache=200), -1.0)

    def test_score_answer_combines_correctness_and_length(self):
        module = self.load()
        correct = module.score_answer(
            "reasoning \\boxed{72}",
            "72",
            completion_tokens=900,
            max_tokens=1000,
            overlong_cache=200,
            enable_overlong=True,
        )
        self.assertTrue(correct.correct)
        self.assertEqual(correct.base_reward, 1.0)
        self.assertAlmostEqual(correct.shaped_reward, 0.5)

        wrong = module.score_answer(
            "\\boxed{71}",
            "72",
            completion_tokens=100,
            max_tokens=1000,
            overlong_cache=200,
            enable_overlong=False,
        )
        self.assertFalse(wrong.correct)
        self.assertEqual(wrong.shaped_reward, -1.0)

    def test_overlong_penalty_can_be_disabled(self):
        module = self.load()
        result = module.score_answer(
            "\\boxed{72}",
            "72",
            completion_tokens=1000,
            max_tokens=1000,
            overlong_cache=200,
            enable_overlong=False,
        )
        self.assertEqual(result.length_penalty, 0.0)
        self.assertEqual(result.shaped_reward, 1.0)


if __name__ == "__main__":
    unittest.main()
