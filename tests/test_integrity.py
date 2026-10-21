import io
import json
import urllib.error
import zipfile

import pytest

from edge_agent_lab.common import write_json
from edge_agent_lab.data import download, ensure_data


class Response(io.BytesIO):
    pass


def test_failed_download_never_promotes_partial_file(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(urllib.error.URLError):
        download(tmp_path)
    assert not (tmp_path / "uci_har.zip").exists()
    assert not (tmp_path / "uci_har.part").exists()


def test_html_error_page_is_not_accepted_as_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(b"<html>error</html>"))
    with pytest.raises(ValueError, match="not a ZIP"):
        download(tmp_path)
    assert not (tmp_path / "uci_har.zip").exists()


def test_source_identity_is_pinned_before_parsing(tmp_path):
    with zipfile.ZipFile(tmp_path / "uci_har.zip", "w") as archive:
        archive.writestr("not-the-dataset.txt", "wrong source")
    with pytest.raises(ValueError, match="checksum mismatch"):
        download(tmp_path)


def test_cached_manifest_cannot_change_source_identity(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"archive_sha256": "wrong"}))
    with pytest.raises(ValueError, match="source identity"):
        ensure_data(tmp_path)


def test_atomic_json_preserves_previous_artifact_when_serialization_fails(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"valid": True})
    with pytest.raises(ValueError):
        write_json(path, {"accuracy": float("nan")})
    assert json.loads(path.read_text()) == {"valid": True}
    assert list(tmp_path.iterdir()) == [path]
