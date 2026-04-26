import json
import os
from pathlib import Path

import pytest

from artifact_provenance import (
    build_provenance,
    collect_git_commit,
    sha256_file,
    write_json_with_provenance,
)


def test_sha256_file_matches_for_identical_content_and_differs_for_different(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    c = tmp_path / "c.txt"
    a.write_bytes(b"hello world")
    b.write_bytes(b"hello world")
    c.write_bytes(b"different bytes")

    assert sha256_file(a) == sha256_file(b)
    assert sha256_file(a) != sha256_file(c)
    assert len(sha256_file(a)) == 64


def test_build_provenance_returns_required_keys(tmp_path):
    input_a = tmp_path / "in_a.json"
    input_b = tmp_path / "in_b.json"
    input_a.write_text("{}")
    input_b.write_text('{"x": 1}')

    argv = ["script.py", "--flag", "value"]
    provenance = build_provenance(
        input_paths=[input_a, input_b],
        argv=argv,
        seed=42,
        schema_version="2",
        repo_root=tmp_path,
    )

    expected_keys = {
        "created_at_utc",
        "schema_version",
        "argv",
        "input_files",
        "git_commit",
        "random_seed",
    }
    assert expected_keys.issubset(provenance.keys())
    assert provenance["argv"] == argv
    assert provenance["argv"] is not argv
    assert provenance["random_seed"] == 42
    assert provenance["schema_version"] == "2"
    assert len(provenance["input_files"]) == 2
    for entry, src in zip(provenance["input_files"], [input_a, input_b]):
        assert entry["path"] == str(src)
        assert entry["sha256"] == sha256_file(src)
    assert provenance["git_commit"] is None


def test_build_provenance_defaults_seed_and_repo(tmp_path):
    src = tmp_path / "in.txt"
    src.write_bytes(b"abc")
    provenance = build_provenance(input_paths=[src], argv=["s.py"])
    assert provenance["random_seed"] is None
    assert provenance["git_commit"] is None


def test_collect_git_commit_returns_none_outside_repo(tmp_path):
    assert collect_git_commit(tmp_path) is None


def test_write_json_with_provenance_writes_valid_json_and_is_atomic(tmp_path):
    src = tmp_path / "in.txt"
    src.write_bytes(b"payload bytes")
    provenance = build_provenance(
        input_paths=[src], argv=["s.py"], seed=7, schema_version="1"
    )
    output = tmp_path / "out" / "result.json"
    payload = {"results": [1, 2, 3], "summary": {"ok": True}}

    write_json_with_provenance(output, payload, provenance)

    assert output.exists()
    document = json.loads(output.read_text())
    assert document["results"] == [1, 2, 3]
    assert document["summary"] == {"ok": True}
    assert document["provenance"]["random_seed"] == 7
    assert document["provenance"]["input_files"][0]["sha256"] == sha256_file(src)

    leftover = [p for p in output.parent.iterdir() if p.name != output.name]
    assert leftover == []


def test_write_json_with_provenance_respects_umask_for_new_file(tmp_path):
    output = tmp_path / "out.json"
    previous_umask = os.umask(0o022)
    try:
        write_json_with_provenance(output, {"x": 1}, {"schema_version": "1"})
    finally:
        os.umask(previous_umask)
    mode = output.stat().st_mode & 0o777
    assert mode == 0o644, f"expected 0o644 under umask 0o022, got {oct(mode)}"


def test_write_json_with_provenance_preserves_existing_file_mode(tmp_path):
    output = tmp_path / "out.json"
    output.write_text("{}")
    os.chmod(output, 0o640)
    write_json_with_provenance(output, {"x": 2}, {"schema_version": "1"})
    mode = output.stat().st_mode & 0o777
    assert mode == 0o640, f"expected preserved 0o640, got {oct(mode)}"


def test_write_json_with_provenance_rejects_existing_provenance_key(tmp_path):
    output = tmp_path / "out.json"
    with pytest.raises(ValueError):
        write_json_with_provenance(
            output, {"provenance": "nope"}, {"schema_version": "1"}
        )
    assert not output.exists()
