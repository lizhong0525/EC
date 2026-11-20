from pathlib import Path

from scripts.secret_scan import scan


def test_secret_scan_detects_likely_key(tmp_path: Path) -> None:
    unsafe_value = "sk-" + "example_key_material_1234567890"
    (tmp_path / "unsafe.py").write_text(
        f"API_KEY='{unsafe_value}'\n",
        encoding="utf-8",
    )
    assert scan(tmp_path) == [(Path("unsafe.py"), 1)]


def test_secret_scan_ignores_placeholder(tmp_path: Path) -> None:
    (tmp_path / ".env.example").write_text("LITELLM_API_KEY=\n", encoding="utf-8")
    assert scan(tmp_path) == []


def test_secret_scan_ignores_local_research_environment(tmp_path: Path) -> None:
    dependency = tmp_path / ".research-venv" / "dependency.py"
    dependency.parent.mkdir()
    dependency.write_text("API_" + "KEY='dependency_fixture_not_project_12345'\n", encoding="utf-8")
    assert scan(tmp_path) == []
