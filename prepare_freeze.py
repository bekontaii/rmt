"""USER-run prospective sealing and registration of a REAL Moodle upload.

prepare: before upload, seals config/data/code/Word/preflight; does not claim an
upload. register: only after the user uploads the Word document, records their
Moodle timestamp and copies their actual receipt file unchanged. Neither command
contacts Moodle, loads a model, or generates final outcomes.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import shutil
import sys
from zipfile import ZipFile, BadZipFile

import freeze_gate as gate


def write_new_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def copy_unchanged(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    gate.require(source.is_file() and source.stat().st_size > 0, "Missing or empty source artifact: " + str(source))
    if target.exists():
        gate.require(target.is_file() and gate.sha256(source) == gate.sha256(target),
                     "Refusing to overwrite a different artifact: " + str(target))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    gate.require(gate.sha256(source) == gate.sha256(target), "Copied artifact bytes differ")
    return target


def prepare(root, protocol_path, preflight_path):
    root = Path(root).resolve()
    seal_path, upload_path = gate.inside(root, gate.SEAL_PATH), gate.inside(root, gate.UPLOAD_PATH)
    gate.require(not seal_path.exists() and not upload_path.exists(), "A seal/upload record already exists; do not overwrite frozen evidence")
    config_path = root / "experiment_config.json"
    cfg = gate.read_json(config_path)
    gate.require(cfg.get("status") in {"draft", "frozen"}, "Expected a draft configuration (or an unregistered preparation retry)")
    inputs = gate.normalize_inputs(root, cfg, gate.expected_input_paths(root, cfg))
    sources = gate.check_sources(root, gate.REQUIRED_SOURCES)
    analysis = gate.read_json(root / "analysis_plan.json")
    gate.require(bool(analysis), "Complete the analysis plan before sealing")
    preflight_path = gate.inside(root, preflight_path)
    snapshot_path = preflight_path.parent / "config_snapshot.json"
    report = gate.verify_preflight(root, cfg, inputs, preflight_path, snapshot_path)
    protocol_path = Path(protocol_path).expanduser().resolve()
    gate.require(protocol_path.is_file(), "The completed Protocol Freeze Word file is required")
    try:
        with ZipFile(protocol_path) as archive:
            gate.require("word/document.xml" in archive.namelist(), "Expected a real DOCX protocol")
    except BadZipFile as exc:
        raise gate.FreezeError("Protocol must be a valid DOCX file") from exc

    # Nothing below generates outcomes or records an upload. Copies retain bytes.
    canonical = copy_unchanged(protocol_path, root / gate.PROTOCOL_PATH)
    snapshots = root / "protocol/preflight"
    copied_report = copy_unchanged(preflight_path, snapshots / "report.json")
    copied_config = copy_unchanged(snapshot_path, snapshots / "config_snapshot.json")
    source_snapshots = {}
    for name in gate.PREFLIGHT_SOURCES:
        source = preflight_path.parent / "source_snapshot" / name
        gate.require(source.is_file() and gate.sha256(source) == report["source_sha256"][name],
                     "Original preflight source snapshot is missing or changed: " + name)
        target = copy_unchanged(source, snapshots / "source_snapshot" / name)
        source_snapshots[name] = gate.file_record(root, target)
    contents = gate.resource_bytes()
    resource_snapshots = {}
    for name in gate.RESOURCE_NAMES:
        target = snapshots / "nltk_english" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            gate.require(target.read_bytes() == contents[name], "An existing Punkt snapshot differs: " + name)
        else:
            with target.open("xb") as stream:
                stream.write(contents[name])
        resource_snapshots[name] = gate.file_record(root, target)

    frozen = dict(cfg, status="frozen")
    gate.require(gate.read_json(config_path) == cfg, "Configuration changed during preparation")
    gate.require(gate.check_sources(root, gate.REQUIRED_SOURCES) == sources, "Code or analysis changed during preparation")
    frozen_text = json.dumps(frozen, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temp_config = config_path.with_name(".experiment_config.freezing.json")
    gate.require(not temp_config.exists(), "A previous temporary config exists; inspect it before retrying")
    with temp_config.open("x", encoding="utf-8") as stream:
        stream.write(frozen_text)
    temp_config.replace(config_path)
    now = datetime.now(timezone.utc).isoformat()
    packages = dict(report["package_versions"])
    packages.update(gate.package_versions(["torch"]))
    plan = {
        "schema_version": 1, "status": "sealed_not_uploaded", "sealed_at_utc": now,
        "inputs": {name: gate.file_record(root, path) for name, path in inputs.items()},
        "source_sha256": sources, "protocol": gate.file_record(root, canonical),
        "preflight": {
            "report": gate.file_record(root, copied_report),
            "config_snapshot": gate.file_record(root, copied_config),
            "source_snapshot": source_snapshots, "resource_snapshot": resource_snapshots,
        },
        "environment": {
            "python_version": report["python_version"], "package_versions": packages,
            "nltk_english_resource_sha256": gate.resource_hashes(contents),
        },
        "upload_status": "not_registered",
        "preparation_note": "Only configuration status changed to frozen; this record does not assert a Moodle upload or instructor approval",
    }
    gate._verify_plan(root, frozen, inputs, gate.REQUIRED_SOURCES, plan)
    write_new_json(seal_path, plan)
    return {"status": "sealed_not_uploaded", "sealed_plan": gate.relative(root, seal_path),
            "sealed_plan_sha256": gate.sha256(seal_path), "protocol": gate.PROTOCOL_PATH,
            "protocol_sha256": plan["protocol"]["sha256"],
            "next_step": "Upload the sealed Word document to Moodle; then register its actual Moodle timestamp and receipt file"}


def register(root, uploaded_at, receipt_path):
    root = Path(root).resolve()
    seal_path, upload_path = root / gate.SEAL_PATH, root / gate.UPLOAD_PATH
    gate.require(not upload_path.exists(), "An upload record already exists; refusing to overwrite it")
    plan = gate.read_json(seal_path)
    cfg = gate.read_json(root / "experiment_config.json")
    inputs = gate.expected_input_paths(root, cfg)
    gate._verify_plan(root, cfg, inputs, gate.REQUIRED_SOURCES, plan)
    upload_time = gate.parse_timestamp(uploaded_at, "actual Moodle uploaded_at")
    seal_time = gate.parse_timestamp(plan["sealed_at_utc"], "sealed_at_utc")
    now = datetime.now(timezone.utc)
    gate.require(seal_time - timedelta(seconds=gate.UPLOAD_PRECISION_TOLERANCE_SECONDS) <= upload_time <= now,
                 "Use the actual Moodle upload time after sealing; minute-resolution tolerance is 60 seconds; future timestamps are invalid")
    receipt_path = Path(receipt_path).expanduser().resolve()
    gate.require(receipt_path.is_file() and receipt_path.stat().st_size > 0, "Provide the actual non-empty Moodle receipt file")
    receipt_hash = gate.sha256(receipt_path)
    protected_hashes = {record["sha256"] for record in plan["inputs"].values()} | set(plan["source_sha256"].values()) | {
        gate.sha256(seal_path), plan["protocol"]["sha256"], plan["preflight"]["report"]["sha256"],
        plan["preflight"]["config_snapshot"]["sha256"]}
    gate.require(receipt_hash not in protected_hashes, "A protocol, dataset, config, source or preflight file is not an upload receipt")
    suffix = receipt_path.suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
        suffix = ".bin"
    copied = copy_unchanged(receipt_path, root / "protocol/upload_receipts" / (receipt_hash + suffix))
    record = {
        "schema_version": 1, "status": "uploaded", "evidence_basis": "user_provided_moodle_receipt",
        "user_attested_upload": True, "uploaded_at": uploaded_at.strip(),
        "registered_at_utc": now.isoformat(), "sealed_plan_sha256": gate.sha256(seal_path),
        "protocol_sha256": plan["protocol"]["sha256"], "receipt": gate.file_record(root, copied),
        "timestamp_precision_tolerance_seconds": gate.UPLOAD_PRECISION_TOLERANCE_SECONDS,
        "verification_scope": "The user supplied the Moodle timestamp and receipt; local integrity checks do not authenticate the Moodle server",
    }
    write_new_json(upload_path, record)
    return gate.load_freeze(root, cfg, inputs, gate.REQUIRED_SOURCES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    actions = parser.add_subparsers(dest="action", required=True)
    before = actions.add_parser("prepare", help="Seal the completed prospective plan BEFORE Moodle upload")
    before.add_argument("--protocol", type=Path, required=True)
    before.add_argument("--preflight", type=Path, required=True, help="Passed preflight report.json, inside the project")
    after = actions.add_parser("register", help="Record the user's real upload AFTER uploading the sealed Word document")
    after.add_argument("--uploaded-at", required=True, help="Actual Moodle ISO8601 time with UTC offset")
    after.add_argument("--receipt", type=Path, required=True, help="Actual local Moodle receipt file; copied unchanged")
    args = parser.parse_args()
    try:
        result = prepare(args.root, args.protocol, args.preflight) if args.action == "prepare" else register(args.root, args.uploaded_at, args.receipt)
    except (gate.FreezeError, OSError, KeyError, TypeError, ValueError) as exc:
        print("LOCKED: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
