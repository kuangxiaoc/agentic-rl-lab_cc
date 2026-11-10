"""Dependency-free tests for the shared local GPU trainer helpers.

Torch, transformers and peft are imported inside the helper functions, so these
tests check the sampling contract with small stand-ins instead of a GPU.
"""

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import local_gpu_trainer as trainer  # noqa: E402


EOS = 99
PROMPT = [1, 2, 3]
GROUP_COMPLETIONS = [[7, 8, EOS, 5], [10, 11]]


class _Value:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Logits:
    def __init__(self, base):
        self.base = base

    def float(self):
        return self

    def __getitem__(self, token):
        return _Value(self.base + token)


class _ScoreStep:
    """`scores[position][batch_index]` so each sampled token gets a unique value."""

    def __init__(self, position):
        self.position = position

    def __getitem__(self, index):
        return _Logits(self.position * 100 + index)


class _Row:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, key):
        return _Row(self.values[key]) if isinstance(key, slice) else self.values[key]

    def __len__(self):
        return len(self.values)

    def tolist(self):
        return list(self.values)


class _Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __getitem__(self, key):
        return _Row(self.rows[key]) if isinstance(key, int) else self.rows[key]


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeTorch:
    def __init__(self):
        self.seeds = []
        self.cuda = type("Cuda", (), {"empty_cache": lambda self: None})()

    def manual_seed(self, value):
        self.seeds.append(value)

    def no_grad(self):
        return _NoGrad()

    def log_softmax(self, logits, dim=-1):
        return logits


class FakeOutput:
    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores


class FakeModel:
    def __init__(self):
        self.calls = []
        self.training = True
        self.training_when_generating = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        self.training_when_generating.append(self.training)
        prompt = kwargs["input_ids"][0].tolist()
        group = kwargs["num_return_sequences"]
        rows = [
            prompt + GROUP_COMPLETIONS[index % len(GROUP_COMPLETIONS)]
            for index in range(group)
        ]
        scores = [_ScoreStep(step) for step in range(max(len(row) - len(prompt) for row in rows))]
        return FakeOutput([_Row(row) for row in rows], scores)

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self


class FakeTokenizer:
    def __init__(self, pad_token_id=None, eos_token_id=EOS, chat_template=None):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        if chat_template is not None:
            self.chat_template = chat_template

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(str(token) for token in tokens)


class FakeTemplateTokenizer(FakeTokenizer):
    def __init__(self, chat_template=None):
        super().__init__(chat_template=chat_template)
        self.calls = []

    def apply_chat_template(self, *messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {
            "input_ids": _Matrix([PROMPT]),
            "attention_mask": _Matrix([[1] * len(PROMPT)]),
            "token_type_ids": _Matrix([[0] * len(PROMPT)]),
        }


class ChatTemplateTests(unittest.TestCase):
    def test_conversation_is_passed_positionally(self):
        tokenizer = FakeTemplateTokenizer()
        trainer.apply_chat_template(tokenizer, [{"role": "user", "content": "hi"}])
        (messages, kwargs), = tokenizer.calls
        self.assertEqual(messages[0][0]["content"], "hi")
        self.assertNotIn("conversation", kwargs)
        self.assertNotIn("messages", kwargs)
        self.assertTrue(kwargs["return_dict"])
        self.assertEqual(kwargs["return_tensors"], "pt")

    def test_token_type_ids_are_dropped(self):
        inputs = trainer.apply_chat_template(FakeTemplateTokenizer(), [])
        self.assertNotIn("token_type_ids", inputs)
        self.assertIn("input_ids", inputs)

    def test_enable_thinking_only_when_the_template_reads_it(self):
        without_flag = FakeTemplateTokenizer(chat_template="please answer")
        trainer.apply_chat_template(without_flag, [], vision=False)
        self.assertNotIn("enable_thinking", without_flag.calls[0][1])

        with_flag = FakeTemplateTokenizer(chat_template="{% if enable_thinking %}x{% endif %}")
        trainer.apply_chat_template(with_flag, [], vision=False)
        self.assertIs(with_flag.calls[0][1]["enable_thinking"], False)

        vision = FakeTemplateTokenizer(chat_template="{% if enable_thinking %}x{% endif %}")
        trainer.apply_chat_template(vision, [], vision=True)
        self.assertNotIn("enable_thinking", vision.calls[0][1])

    def test_chat_template_mentions_handles_every_template_shape(self):
        self.assertFalse(trainer.chat_template_mentions(object(), "enable_thinking"))
        self.assertFalse(trainer.chat_template_mentions(FakeTokenizer(), "enable_thinking"))
        by_task = FakeTokenizer(chat_template={"default": "no", "video": "enable_thinking"})
        self.assertTrue(trainer.chat_template_mentions(by_task, "enable_thinking"))

    def test_pad_token_id_falls_back_to_eos(self):
        self.assertEqual(trainer.pad_token_id(FakeTokenizer(pad_token_id=0)), 0)
        self.assertEqual(trainer.pad_token_id(FakeTokenizer()), EOS)
        self.assertIsNone(trainer.pad_token_id(object()))

    def test_processor_inner_tokenizer_is_unwrapped(self):
        """Qwen3VLProcessor wraps the tokenizer and has no eos_token_id itself."""
        inner = FakeTokenizer(pad_token_id=0)

        class FakeProcessor:
            tokenizer = inner

        processor = FakeProcessor()
        self.assertFalse(hasattr(processor, "eos_token_id"))
        self.assertIs(trainer.text_tokenizer(processor), inner)
        self.assertEqual(trainer.pad_token_id(processor), 0)
        self.assertIs(trainer.text_tokenizer(inner), inner)


class SupportedKwargsTests(unittest.TestCase):
    def test_unknown_kwargs_are_dropped(self):
        def known(cache_dir, allow_patterns=None):
            return cache_dir, allow_patterns

        self.assertEqual(
            trainer.supported_kwargs(known, cache_dir="c", allow_patterns=["p"], nope=1),
            {"cache_dir": "c", "allow_patterns": ["p"]},
        )

    def test_var_keyword_functions_keep_everything(self):
        def catch_all(**kwargs):
            return kwargs

        self.assertEqual(trainer.supported_kwargs(catch_all, anything=1), {"anything": 1})


class ParquetRowLimitTests(unittest.TestCase):
    def test_parquet_loader_limits_before_materializing_rows(self):
        """Large DAPO parquet must not be copied into Python as 1.7M dicts."""
        import sys
        import tempfile
        import types
        from unittest.mock import patch

        calls = {}

        class Dataset:
            def __init__(self, rows):
                self.rows = rows

            def select(self, indices):
                calls["selected"] = list(indices)
                return Dataset([self.rows[index] for index in indices])

            def shuffle(self, seed):
                calls["seed"] = seed
                return self

            def __iter__(self):
                return iter(self.rows)

            def __len__(self):
                return len(self.rows)

        fake = Dataset([{"id": index} for index in range(10)])
        datasets = types.ModuleType("datasets")
        datasets.load_dataset = lambda *args, **kwargs: fake
        with tempfile.NamedTemporaryFile(suffix=".parquet") as handle, patch.dict(
            sys.modules, {"datasets": datasets}
        ):
            rows = trainer.load_parquet_rows(Path(handle.name), limit=3, seed=7)
        self.assertEqual(calls, {"selected": [0, 1, 2], "seed": 7})
        self.assertEqual(rows, [{"id": 0}, {"id": 1}, {"id": 2}])


class GenerateGroupTests(unittest.TestCase):
    def run_group(self, group_size, tokenizer=None):
        torch_module = FakeTorch()
        model = FakeModel()
        inputs = {
            "input_ids": _Matrix([PROMPT]),
            "attention_mask": _Matrix([[1] * len(PROMPT)]),
        }
        result = trainer.generate_group(
            model,
            tokenizer or FakeTokenizer(),
            inputs,
            group_size=group_size,
            max_new_tokens=8,
            temperature=1.0,
            top_p=1.0,
            seed=42,
            torch_module=torch_module,
            device="cuda:0",
        )
        return model.calls[0], torch_module, result

    def test_one_prompt_generates_exactly_group_size_completions(self):
        call, torch_module, (completions, old_logprobs, texts, prompt_length) = self.run_group(2)
        self.assertEqual(call["num_return_sequences"], 2)
        self.assertEqual(call["pad_token_id"], EOS)
        self.assertNotIn("eos_token_id", call)
        self.assertEqual(torch_module.seeds, [42])
        self.assertEqual(prompt_length, len(PROMPT))
        self.assertEqual(len(completions), 2)
        self.assertEqual(len(texts), 2)

    def test_inputs_reach_generate_without_manual_repeat(self):
        call, _, _ = self.run_group(2)
        self.assertEqual(call["input_ids"].shape, (1, len(PROMPT)))
        self.assertEqual(call["attention_mask"].shape, (1, len(PROMPT)))

    def test_completions_stop_at_eos_and_keep_matching_logprobs(self):
        _, _, (completions, old_logprobs, _, _) = self.run_group(2)
        self.assertEqual(completions[0], [7, 8, EOS])
        self.assertEqual(completions[1], [10, 11])
        self.assertEqual([len(row) for row in old_logprobs], [3, 2])
        self.assertEqual(old_logprobs[0], [7, 108, 299])

    def test_larger_group_stays_one_group(self):
        call, _, (completions, _, _, _) = self.run_group(4)
        self.assertEqual(call["num_return_sequences"], 4)
        self.assertEqual(len(completions), 4)

    def test_sampling_runs_in_eval_mode_and_restores_train_mode(self):
        """Gradient checkpointing disables the KV cache, so sampling must use eval mode."""
        model = FakeModel()
        model.train()
        inputs = {"input_ids": _Matrix([PROMPT])}
        trainer.generate_group(
            model,
            FakeTokenizer(),
            inputs,
            group_size=2,
            max_new_tokens=8,
            temperature=1.0,
            top_p=1.0,
            seed=1,
            torch_module=FakeTorch(),
            device="cuda:0",
        )
        self.assertEqual(model.training_when_generating, [False])
        self.assertTrue(model.training, "model must be returned in train mode for the loss step")

    def test_a_model_already_in_eval_mode_is_left_alone(self):
        model = FakeModel()
        model.eval()
        trainer.generate_group(
            model,
            FakeTokenizer(),
            {"input_ids": _Matrix([PROMPT])},
            group_size=1,
            max_new_tokens=8,
            temperature=1.0,
            top_p=1.0,
            seed=1,
            torch_module=FakeTorch(),
            device="cuda:0",
        )
        self.assertFalse(model.training)

    def test_processor_is_accepted_for_eos_and_decoding(self):
        inner = FakeTokenizer()

        class FakeProcessor:
            tokenizer = inner

        call, _, (completions, _, texts, _) = self.run_group(2, tokenizer=FakeProcessor())
        self.assertEqual(call["pad_token_id"], EOS)
        self.assertEqual(completions[0], [7, 8, EOS])
        self.assertEqual(len(texts), 2)


if __name__ == "__main__":
    unittest.main()
