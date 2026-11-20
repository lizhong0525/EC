from pathlib import Path

import pytest

from ecocompanion.experiments import io


def test_atomic_write_retries_transient_windows_read_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "checkpoint.json"
    io.write_json_atomic(path, {"step": 0}, overwrite=False)
    original = io.os.replace
    calls = 0

    def temporarily_locked(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            error = PermissionError("temporary Windows reader")
            error.winerror = 32
            raise error
        original(source, destination)

    monkeypatch.setattr(io.os, "replace", temporarily_locked)
    monkeypatch.setattr(io.time, "sleep", lambda _: None)
    io.write_json_atomic(path, {"step": 1}, overwrite=True)
    assert calls == 3
    assert io.read_json_object(path) == {"step": 1}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("windows_error", [None, 5])
def test_permanent_failure_keeps_original_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows_error: int | None
) -> None:
    path = tmp_path / "checkpoint.json"
    io.write_json_atomic(path, {"step": 0}, overwrite=False)
    calls = 0

    def denied(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        error = PermissionError("permanent failure")
        if windows_error is not None:
            error.winerror = windows_error
        raise error

    monkeypatch.setattr(io.os, "replace", denied)
    monkeypatch.setattr(io.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        io.write_json_atomic(path, {"step": 1}, overwrite=True)
    assert calls == (1 if windows_error is None else 8)
    assert io.read_json_object(path) == {"step": 0}
    assert not list(tmp_path.glob("*.tmp"))
