"""Validate a downloaded P1 JQ index-input tar.gz or extracted directory."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


DEFAULT_MAX_MEMBER_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024


def _is_safe_member_name(name):
    path = PurePosixPath(name)
    return (
        bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in name
    )


def _read_regular_member(archive, member, max_member_bytes):
    if not member.isfile():
        raise ValueError("archive member is not a regular file: %s" % member.name)
    if member.size > max_member_bytes:
        raise ValueError("archive member exceeds size limit: %s" % member.name)
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError("cannot read archive member: %s" % member.name)
    content = extracted.read(max_member_bytes + 1)
    if len(content) != member.size:
        raise ValueError("archive member size mismatch: %s" % member.name)
    return content


def _validate_manifest_header(manifest):
    archive_root = str(manifest.get("archive_root", ""))
    dataset_id = str(manifest.get("dataset_id", ""))
    if not archive_root or archive_root != dataset_id:
        raise ValueError("manifest archive_root must equal dataset_id")
    if manifest.get("archive_format") != "tar+gzip":
        raise ValueError("manifest archive_format is not tar+gzip")

    file_records = manifest.get("files")
    if not isinstance(file_records, list) or not file_records:
        raise ValueError("manifest files must be a non-empty list")
    records_by_path = {}
    for record in file_records:
        path = record.get("path")
        if not isinstance(path, str) or not _is_safe_member_name(path):
            raise ValueError("manifest contains an unsafe file path")
        if path in records_by_path:
            raise ValueError("manifest contains duplicate file records")
        if not path.startswith(archive_root + "/data/"):
            raise ValueError("payload is outside the archive data directory")
        records_by_path[path] = record
    return archive_root, dataset_id, records_by_path


def _validate_payload_content(path, content, record):
    if int(record.get("bytes", -1)) != len(content):
        raise ValueError("byte count mismatch: %s" % path)
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != record.get("sha256"):
        raise ValueError("sha256 mismatch: %s" % path)


def validate_p1_index_archive(
    archive_path,
    max_member_bytes=DEFAULT_MAX_MEMBER_BYTES,
    max_total_bytes=DEFAULT_MAX_TOTAL_BYTES,
):
    """Check archive membership, safe paths, byte counts and checksums."""

    archive_path = Path(archive_path).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(str(archive_path))
    if not archive_path.name.endswith(".tar.gz"):
        raise ValueError("expected a .tar.gz archive")

    archive_digest = hashlib.sha256()
    with archive_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            archive_digest.update(block)

    with tarfile.open(str(archive_path), mode="r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError("archive contains no members")
        if any(not _is_safe_member_name(member.name) for member in members):
            raise ValueError("archive contains an unsafe member path")
        if len(set(member.name for member in members)) != len(members):
            raise ValueError("archive contains duplicate member paths")
        total_bytes = sum(member.size for member in members)
        if total_bytes > max_total_bytes:
            raise ValueError("archive uncompressed size exceeds safety limit")

        manifest_members = [
            member for member in members if member.name.endswith("/manifest.json")
        ]
        if len(manifest_members) != 1:
            raise ValueError("archive must contain exactly one root manifest.json")
        manifest_content = _read_regular_member(
            archive, manifest_members[0], max_member_bytes
        )
        manifest = json.loads(manifest_content.decode("utf-8"))

        archive_root, dataset_id, records_by_path = _validate_manifest_header(
            manifest
        )
        if manifest_members[0].name != "%s/manifest.json" % archive_root:
            raise ValueError("manifest path does not match archive_root")

        members_by_path = {member.name: member for member in members}
        expected_paths = set(records_by_path)
        expected_paths.add(manifest_members[0].name)
        if set(members_by_path) != expected_paths:
            raise ValueError("archive members do not exactly match manifest files")

        for path in sorted(records_by_path):
            record = records_by_path[path]
            content = _read_regular_member(
                archive, members_by_path[path], max_member_bytes
            )
            _validate_payload_content(path, content, record)

    return {
        "path": str(archive_path),
        "dataset_id": dataset_id,
        "archive_sha256": archive_digest.hexdigest(),
        "members": len(members),
        "payload_files": len(records_by_path),
        "uncompressed_bytes": total_bytes,
        "manifest": manifest,
    }


def validate_p1_index_directory(
    snapshot_dir,
    max_member_bytes=DEFAULT_MAX_MEMBER_BYTES,
    max_total_bytes=DEFAULT_MAX_TOTAL_BYTES,
):
    """Validate an extracted snapshot against its unchanged JQ manifest."""

    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    if not snapshot_dir.is_dir():
        raise NotADirectoryError(str(snapshot_dir))
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("snapshot directory must contain a regular manifest.json")
    manifest_content = manifest_path.read_bytes()
    if len(manifest_content) > max_member_bytes:
        raise ValueError("manifest exceeds size limit")
    manifest = json.loads(manifest_content.decode("utf-8"))
    archive_root, dataset_id, records_by_path = _validate_manifest_header(manifest)
    if snapshot_dir.name != archive_root:
        raise ValueError("snapshot directory name does not match archive_root")

    expected_relative_paths = {"manifest.json"}
    for path in records_by_path:
        relative_path = PurePosixPath(path).relative_to(archive_root).as_posix()
        expected_relative_paths.add(relative_path)
    actual_entries = list(snapshot_dir.rglob("*"))
    if any(entry.is_symlink() for entry in actual_entries):
        raise ValueError("snapshot directory contains a symbolic link")
    actual_relative_paths = {
        entry.relative_to(snapshot_dir).as_posix()
        for entry in actual_entries
        if entry.is_file()
    }
    if actual_relative_paths != expected_relative_paths:
        raise ValueError("snapshot files do not exactly match manifest files")

    total_bytes = len(manifest_content)
    for path in sorted(records_by_path):
        record = records_by_path[path]
        relative_path = PurePosixPath(path).relative_to(archive_root).as_posix()
        payload_path = snapshot_dir / relative_path
        size = payload_path.stat().st_size
        if size > max_member_bytes:
            raise ValueError("snapshot file exceeds size limit: %s" % relative_path)
        total_bytes += size
        if total_bytes > max_total_bytes:
            raise ValueError("snapshot uncompressed size exceeds safety limit")
        content = payload_path.read_bytes()
        _validate_payload_content(path, content, record)

    return {
        "path": str(snapshot_dir),
        "dataset_id": dataset_id,
        "manifest_sha256": hashlib.sha256(manifest_content).hexdigest(),
        "members": len(actual_relative_paths),
        "payload_files": len(records_by_path),
        "uncompressed_bytes": total_bytes,
        "manifest": manifest,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate one P1 JQ index-input tar.gz or extracted directory"
    )
    parser.add_argument("source", help="path to the .tar.gz or extracted directory")
    args = parser.parse_args(argv)
    source = Path(args.source).expanduser()
    if source.is_dir():
        result = validate_p1_index_directory(source)
        print("valid extracted snapshot: %s" % result["path"])
        print("manifest sha256: %s" % result["manifest_sha256"])
    else:
        result = validate_p1_index_archive(source)
        print("valid archive: %s" % result["path"])
        print("archive sha256: %s" % result["archive_sha256"])
    print("dataset: %s" % result["dataset_id"])
    print("members: %d" % result["members"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
