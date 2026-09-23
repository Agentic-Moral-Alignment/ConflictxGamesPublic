"""Coverage for Dropbox-safe atomic replaces used by Message Batch state."""

from pathlib import Path

from conflictxgame.common.portable_lock import replace_with_retry


def test_replace_with_retry_succeeds_after_permission_errors(monkeypatch, tmp_path):
    src = tmp_path / "src.json"
    dst = tmp_path / "dst.json"
    src.write_text("new", encoding="utf-8")
    dst.write_text("old", encoding="utf-8")
    calls = {"n": 0}

    def flaky_replace(source, dest):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("simulated Dropbox lock")
        Path(dest).write_text(Path(source).read_text(encoding="utf-8"), encoding="utf-8")
        Path(source).unlink()

    monkeypatch.setattr(
        "conflictxgame.common.portable_lock.os.replace",
        flaky_replace,
    )
    monkeypatch.setattr(
        "conflictxgame.common.portable_lock.time.sleep",
        lambda _seconds: None,
    )
    replace_with_retry(str(src), str(dst), attempts=5)
    assert calls["n"] == 3
    assert dst.read_text(encoding="utf-8") == "new"
