from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def import_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE_DATA = import_script("wikitext_prepare_data", ROOT / "scripts" / "prepare-data.py")
PREPARE_INIT = import_script("wikitext_prepare_init", ROOT / "scripts" / "prepare-initialization.py")
SHUFFLE = import_script("wikitext_seeded_shuffle", ROOT / "scripts" / "seeded_shuffle.py")


class FakeNLTK:
    @staticmethod
    def sent_tokenize(text: str) -> list[str]:
        return [sentence.strip() for sentence in text.split("|") if sentence.strip()]

    @staticmethod
    def word_tokenize(sentence: str) -> list[str]:
        return sentence.split()


class WikiTextPreparationTests(unittest.TestCase):
    def test_splitmix_epoch_order_is_repeatable_and_is_a_permutation(self) -> None:
        first = SHUFFLE.splitmix64_shuffle_indices(100, 42)
        self.assertEqual(first, SHUFFLE.splitmix64_shuffle_indices(100, 42))
        self.assertNotEqual(first, SHUFFLE.splitmix64_shuffle_indices(100, 43))
        self.assertEqual(sorted(first), list(range(100)))

    def test_sentence_preparation_is_seeded_and_adds_boundaries(self) -> None:
        text = "One word .|Second line !"
        first = PREPARE_DATA.tokenize_text(text, __import__("random").Random(12), FakeNLTK)
        second = PREPARE_DATA.tokenize_text(text, __import__("random").Random(12), FakeNLTK)
        self.assertEqual(first, second)
        self.assertEqual(first.count("<s>"), 2)
        self.assertEqual(first.count("</s>"), 2)
        self.assertNotIn(".", first)
        self.assertNotIn("!", first)
        self.assertTrue(all(token == token.lower() for token in first))

    def test_windows_shift_targets_and_drop_incomplete_tail(self) -> None:
        vocab = {"<unk>": 0, **{f"t{i}": i + 1 for i in range(32)}}
        inputs, targets, count = PREPARE_DATA.sequence_pairs(
            [f"t{i}" for i in range(32)], vocab
        )
        self.assertEqual(count, 1)
        self.assertEqual(inputs, list(range(1, 31)))
        self.assertEqual(targets, list(range(2, 32)))

    def test_unseen_token_maps_to_explicit_unknown_id(self) -> None:
        inputs, targets, count = PREPARE_DATA.sequence_pairs(
            ["train"] * 31, {"<unk>": 0, "train": 1}
        )
        self.assertEqual(count, 1)
        self.assertEqual(inputs, [1] * 30)
        self.assertEqual(targets, [1] * 30)
        inputs, _, _ = PREPARE_DATA.sequence_pairs(
            ["heldout"] * 31, {"<unk>": 0, "train": 1}
        )
        self.assertEqual(inputs, [0] * 30)


class WikiTextInitializationTests(unittest.TestCase):
    def test_initialization_is_reproducible_and_matches_vocab(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            (data / "manifest.json").write_text(json.dumps({
                "vocabulary_size": 17,
                "vocabulary_sha256": "fixture-hash",
            }))
            first, second = root / "first", root / "second"
            PREPARE_INIT.generate(first, data, seed=42)
            PREPARE_INIT.generate(second, data, seed=42)
            manifest = json.loads((first / "manifest.json").read_text())
            self.assertEqual(manifest["vocabulary_size"], 17)
            self.assertEqual(manifest["dataset_vocabulary_sha256"], "fixture-hash")
            for name, metadata in manifest["files"].items():
                self.assertEqual(metadata["sha256"], PREPARE_INIT.sha256_file(first / name))
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            self.assertEqual(manifest["files"]["embedding-weight-f32le.bin"]["shape"], [17, 100])
            self.assertEqual(manifest["files"]["rnn-layer1-weight-ih-f32le.bin"]["shape"], [100, 100])
            self.assertEqual(manifest["files"]["output-weight-f32le.bin"]["shape"], [17, 100])


if __name__ == "__main__":
    unittest.main()
