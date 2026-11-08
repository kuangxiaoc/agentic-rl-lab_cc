import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import modelscope_lib as lib  # noqa: E402


class WorkspaceAndStateTests(unittest.TestCase):
    def test_parse_chapter_specs(self):
        self.assertEqual(lib.parse_chapter_spec("1-3"), [1, 2, 3])
        self.assertEqual(lib.parse_chapter_spec("4-6,8"), [4, 5, 6, 8])
        self.assertEqual(lib.parse_chapter_spec("02"), [2])
        self.assertEqual(lib.parse_chapter_spec("all"), list(range(1, 17)))
        with self.assertRaises(ValueError):
            lib.parse_chapter_spec("3-1")
        with self.assertRaises(ValueError):
            lib.parse_chapter_spec("17")

    def test_state_is_atomic_and_rejects_invalid_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = lib.initialize_state(root)
            self.assertEqual(state["chapters"]["1"]["status"], "not_ported")
            lib.update_chapter_state(root, 1, "completed", "run-1", details={"steps": 1})
            reloaded = lib.load_state(root)
            self.assertEqual(reloaded["chapters"]["1"]["status"], "completed")
            self.assertEqual(reloaded["chapters"]["1"]["details"]["steps"], 1)
            self.assertFalse(list(root.glob("state.json.*")))
            with self.assertRaises(ValueError):
                lib.update_chapter_state(root, 1, "success", "run-1")

    def test_skip_completed_unless_forced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lib.initialize_state(root)
            lib.update_chapter_state(root, 2, "completed", "run-1")
            state = lib.load_state(root)
            self.assertTrue(lib.should_skip_chapter(state, 2, force=False))
            self.assertFalse(lib.should_skip_chapter(state, 2, force=True))

    def test_workspace_root_and_output_path_boundary(self):
        with self.assertRaises(ValueError):
            lib.resolve_workspace_root("/")
        with tempfile.TemporaryDirectory() as tmp:
            root = lib.resolve_workspace_root(tmp)
            self.assertEqual(root, Path(tmp).resolve())
            self.assertTrue(lib.ensure_within_root(root, root / "runs" / "01"))
            with self.assertRaises(ValueError):
                lib.ensure_within_root(root, root.parent / "escape")


class BudgetAndArchiveTests(unittest.TestCase):
    def test_free_token_limit_and_hard_cost_cap(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"ALLOW_PAID_OVERAGE": "0", "MAX_EXTERNAL_COST_CNY": "50"}
        ):
            root = Path(tmp)
            lib.init_budget(root)
            first = lib.record_external_usage(
                root, "bailian", "qwen3.8-flash", 500_000, 100_000
            )
            self.assertEqual(first["models"]["qwen3.8-flash"]["tokens"], 600_000)
            with self.assertRaises(lib.ExternalBudgetError):
                lib.record_external_usage(
                    root, "bailian", "qwen3.8-flash", 300_000, 200_000
                )
            with self.assertRaises(lib.ExternalBudgetError):
                lib.record_external_usage(
                    root,
                    "manual",
                    "priced-api",
                    0,
                    0,
                    estimated_cost_cny=51.0,
                    allow_unlisted_model=True,
                )

    def test_redaction_and_archive_excludes_secrets_and_model_cache(self):
        redacted = lib.redact_text("Bearer sk-abcdefghijklmnop123456 DASHSCOPE_API_KEY=xyz")
        self.assertNotIn("sk-abcdefghijklmnop123456", redacted)
        self.assertNotIn("xyz", redacted)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            repo = Path(tmp) / "repo"
            root.mkdir()
            repo.mkdir()
            (root / ".env").write_text("DASHSCOPE_API_KEY=sk-secret\n", encoding="utf-8")
            (root / "state.json").write_text('{"ok": true}\n', encoding="utf-8")
            (root / "cache").mkdir()
            (root / "cache" / "model.safetensors").write_bytes(b"model")
            run_dir = root / "runs" / "01-loss-functions" / "run-1"
            run_dir.mkdir(parents=True)
            (run_dir / "adapter_model.safetensors").write_bytes(b"adapter")
            (repo / "chapter.py").write_text("print('study')\n", encoding="utf-8")
            (repo / ".env").write_text("SECRET=value\n", encoding="utf-8")
            archive = lib.create_archive(root, repo)
            with tarfile.open(archive, "r:gz") as tar:
                names = tar.getnames()
            joined = "\n".join(names)
            self.assertIn("results/state.json", joined)
            self.assertIn("results/runs/01-loss-functions/run-1/adapter_model.safetensors", joined)
            self.assertIn("code/chapter.py", joined)
            self.assertNotIn(".env", joined)
            self.assertNotIn("cache/model.safetensors", joined)

    def test_report_and_study_notes_are_generated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = lib.initialize_state(root)
            lib.write_study_notes(root)
            lib.write_report(root, state)
            self.assertIn("PyTRIO", (root / "study_notes.md").read_text(encoding="utf-8"))
            self.assertIn("not_ported", (root / "reports" / "latest.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
