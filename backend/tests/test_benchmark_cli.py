import json

from ecocompanion.benchmark.cases import load_cases
from ecocompanion.benchmark.cli import generate_example_dataset, main


def test_generate_example_dataset_writes_cases_and_manifest(tmp_path) -> None:
    destination = generate_example_dataset(
        tmp_path / "example.jsonl",
        seed=17,
        limit=4,
    )

    assert len(load_cases(destination)) == 4
    manifest = json.loads(
        destination.with_suffix(".splits.json").read_text(encoding="utf-8")
    )
    assert manifest["canonical_case_count"] == 144
    assert manifest["exported_case_count"] == 4
    assert set(manifest["cases"].values()) <= {
        "train",
        "dev",
        "public_test",
        "hidden_test",
    }


def test_cli_main_supports_small_export(tmp_path) -> None:
    destination = tmp_path / "from-main.jsonl"

    exit_code = main(["--output", str(destination), "--limit", "2"])

    assert exit_code == 0
    assert len(load_cases(destination)) == 2
