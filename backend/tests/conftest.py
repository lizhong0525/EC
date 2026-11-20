import pytest


@pytest.fixture(autouse=True)
def isolate_messaging_configuration(tmp_path, monkeypatch):
    """Channel tests must never read or overwrite a developer's saved channels."""
    monkeypatch.setenv("ECO_MESSAGING_HOME", str(tmp_path / "test-messaging"))
