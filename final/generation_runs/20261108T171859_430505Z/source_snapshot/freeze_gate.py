"""Read-only integrity gate for a user-attested Moodle protocol upload.

No Moodle server is contacted and no receipt is created by this module. A valid
local receipt record is evidence supplied by the user, not authenticated proof
from Moodle. Importing this module loads no models or tokenizers.
"""
from datetime import datetime, timedelta, timezone
from importlib import metadata
import hashlib
import json
from pathlib import Path
import platform
import re


REQUIRED_SOURCES = (
    "final_retrieval.py", "final_generation.py", "check_final_inputs.py",
    "run_development_retrieval.py", "run_development_generation.py",
    "chunkers.py", "semantic_chunker.py", "sentence_splitter.py",
    "retrieval_metrics.py", "prompt_builder.py", "metrics.py",
    "freeze_gate.py", "prepare_freeze.py",
)
LOCKED_FILES = REQUIRED_SOURCES + (
    "requirements.txt", "requirements.lock.txt", "analysis_plan.json",
)
INPUT_KEYS = ("config", "raw_dataset", "selection_manifest", "corpus", "questions")
RESOURCE_NAMES = (
    "abbrev_types.txt", "collocations.tab", "ortho_context.tab", "sent_starters.txt",
)
PREFLIGHT_PACKAGES = (
    "transformers", "tokenizers", "sentencepiece", "nltk",
    "sentence-transformers", "numpy",
)
PREFLIGHT_SOURCES = (
    "check_final_inputs.py", "chunkers.py", "semantic_chunker.py", "sentence_splitter.py",
)
SEAL_PATH = "protocol/sealed_plan.json"
UPLOAD_PATH = "protocol/freeze_manifest.json"
PROTOCOL_PATH = "protocol/Protocol_Freeze.docx"
UPLOAD_PRECISION_TOLERANCE_SECONDS = 60


class FreezeError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise FreezeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FreezeError("Cannot read the required JSON artifact: " + str(path)) from exc
    require(isinstance(value, dict), "Expected a JSON object: " + str(path))
    return value


def inside(root, path):
    root = Path(root).resolve()
    result = Path(path)
    if not result.is_absolute():
        result = root / result
    result = result.resolve()
    require(root in result.parents, "Artifact path escapes the project: " + str(path))
    return result


def relative(root, path):
    return inside(root, path).relative_to(Path(root).resolve()).as_posix()


def file_record(root, path):
    path = inside(root, path)
    require(path.is_file() and path.stat().st_size > 0, "Missing or empty artifact: " + str(path))
    return {"path": relative(root, path), "sha256": sha256(path)}


def verify_file(root, record, label):
    require(isinstance(record, dict), "Missing artifact record: " + label)
    actual = file_record(root, record.get("path", ""))
    require(actual == record, "Artifact is missing or changed: " + label)
    return inside(root, record["path"])


def parse_timestamp(value, label):
    require(isinstance(value, str) and bool(value.strip()), "Missing timestamp: " + label)
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise FreezeError("Invalid ISO8601 timestamp: " + label) from exc
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
            "Timestamp needs an explicit UTC offset: " + label)
    return parsed.astimezone(timezone.utc)


def expected_input_paths(root, cfg):
    data = cfg["data"]
    return dict(config=inside(root, "experiment_config.json"),
                raw_dataset=inside(root, data["dataset_path"]),
                selection_manifest=inside(root, data["selection_manifest_path"]),
                corpus=inside(root, data["corpus_path"]),
                questions=inside(root, data["questions_path"]))


def normalize_inputs(root, cfg, input_paths):
    require(set(input_paths) == set(INPUT_KEYS), "The five frozen input paths must be supplied")
    expected = expected_input_paths(root, cfg)
    actual = {name: inside(root, input_paths[name]) for name in INPUT_KEYS}
    require(actual == expected, "Runner inputs differ from the configured input paths")
    require(read_json(actual["config"]) == cfg, "Loaded configuration differs from current config bytes")
    return actual


def package_versions(names):
    result = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise FreezeError("Required package is not installed: " + name) from exc
    return result


def resource_bytes():
    try:
        import nltk
        pointer = nltk.data.find("tokenizers/punkt_tab/english/")
        contents = {}
        for name in RESOURCE_NAMES:
            with pointer.join(name).open() as stream:
                contents[name] = stream.read()
    except (ImportError, LookupError, OSError) as exc:
        raise FreezeError("The four English Punkt resources must be installed locally") from exc
    return contents


def resource_hashes(contents=None):
    contents = resource_bytes() if contents is None else contents
    return {name: hashlib.sha256(contents[name]).hexdigest() for name in RESOURCE_NAMES}


def check_sources(root, source_names):
    requested = tuple(source_names)
    require(bool(requested) and all(isinstance(name, str) for name in requested),
            "The runner must declare its inference source names")
    require(set(requested) <= set(REQUIRED_SOURCES),
            "Runner declares an inference module outside the sealed explicit roster")
    return {name: file_record(root, name)["sha256"] for name in LOCKED_FILES}


def verify_preflight(root, cfg, inputs, report_path, snapshot_path):
    """Check existing preflight evidence; do not re-tokenize or evaluate data."""
    report = read_json(report_path)
    require(report.get("purpose") == "final_input_preflight" and report.get("status") == "passed",
            "A passed final-input preflight report is required")
    require(report.get("prediction_runs") == 0 and report.get("model_embedding_runs") == 0,
            "Preflight evidence must report zero model prediction/embedding runs")
    completed = parse_timestamp(report.get("completed_at_utc"), "preflight.completed_at_utc")
    require(completed <= datetime.now(timezone.utc), "Preflight completion is in the future")
    require(report.get("articles_checked") == cfg["data"]["articles"] == 20
            and report.get("retrieval_questions_checked") == cfg["data"]["retrieval_questions"] == 2000
            and report.get("generation_questions_checked") == cfg["data"]["generation_questions"] == 1000,
            "Preflight does not cover the frozen 20/2000/1000 evaluation sets")
    hashes = report.get("input_sha256", {})
    require(set(hashes) == set(INPUT_KEYS), "Preflight input hash list is incomplete")
    for name in INPUT_KEYS[1:]:
        require(sha256(inputs[name]) == hashes[name], "Preflight input changed: " + name)
    require(hashes["raw_dataset"] == cfg["data"]["dataset_sha256"], "Configured dataset hash differs")
    require(sha256(snapshot_path) == hashes["config"], "Preflight config snapshot is missing or changed")
    old_cfg = read_json(snapshot_path)
    require({k: v for k, v in old_cfg.items() if k != "status"}
            == {k: v for k, v in cfg.items() if k != "status"},
            "Configuration changed since preflight beyond draft/frozen status; rerun preflight first")
    sources = report.get("source_sha256", {})
    require(set(sources) == set(PREFLIGHT_SOURCES), "Preflight source hash list is incomplete")
    for name in PREFLIGHT_SOURCES:
        require(sha256(inside(root, name)) == sources[name], "Preflight source changed: " + name)
    packages = report.get("package_versions", {})
    require(isinstance(packages, dict) and set(PREFLIGHT_PACKAGES) <= set(packages),
            "Preflight package metadata is incomplete")
    require(package_versions(packages) == packages, "Package versions differ from preflight")
    require(platform.python_version() == report.get("python_version"), "Python version differs from preflight")
    require(report.get("nltk_english_resource_sha256") == resource_hashes(),
            "English Punkt resources differ from preflight")
    require(report.get("tokenizers") == {
        "embedding": {"name": cfg["embedding"]["name"], "revision": cfg["embedding"]["revision"], "use_fast": True},
        "generation": {"name": cfg["generation"]["name"], "revision": cfg["generation"]["revision"], "use_fast": False, "legacy": True}},
        "Tokenizer configurations differ from preflight")
    stats = report.get("token_stats", {})
    required_stats = {"embedding_questions", "fixed_chunks", "overlap_chunks",
                      "semantic_candidates", "minimum_generation_prompts"}
    require(set(stats) == required_stats, "Preflight token checks are incomplete")
    for name, values in stats.items():
        limit = cfg["generation"]["max_input_tokens"] if name == "minimum_generation_prompts" else cfg["embedding"]["max_input_tokens"]
        require(isinstance(values, dict) and values.get("over_limit") == 0
                and type(values.get("count")) is int and values["count"] > 0
                and type(values.get("max_tokens")) is int and 0 < values["max_tokens"] <= limit,
                "Preflight contains incomplete checks or token overflow: " + name)
    require(stats["embedding_questions"]["count"] == 2000
            and stats["minimum_generation_prompts"]["count"] == 1000,
            "Preflight token-check question counts differ")
    return report


def _verify_plan(root, cfg, input_paths, source_names, plan):
    require(plan.get("schema_version") == 1 and plan.get("status") == "sealed_not_uploaded",
            "Missing prospective sealed plan")
    require(cfg.get("status") == "frozen", "Configuration must be frozen before final inference")
    inputs = normalize_inputs(root, cfg, input_paths)
    require(set(plan.get("inputs", {})) == set(INPUT_KEYS), "Sealed input list is incomplete")
    for name in INPUT_KEYS:
        actual = file_record(root, inputs[name])
        require(actual == plan["inputs"][name], "Frozen input changed: " + name)
    sources = check_sources(root, source_names)
    require(sources == plan.get("source_sha256"), "A required inference source, requirements file or analysis plan changed")
    require(isinstance(read_json(inside(root, "analysis_plan.json")), dict), "Analysis plan must be JSON")
    protocol = verify_file(root, plan.get("protocol"), "Protocol Freeze Word document")
    require(relative(root, protocol) == PROTOCOL_PATH, "Unexpected canonical protocol path")
    preflight = plan.get("preflight", {})
    report_path = verify_file(root, preflight.get("report"), "preflight report snapshot")
    snapshot_path = verify_file(root, preflight.get("config_snapshot"), "preflight config snapshot")
    report = verify_preflight(root, cfg, inputs, report_path, snapshot_path)
    require(set(preflight.get("source_snapshot", {})) == set(PREFLIGHT_SOURCES),
            "Preflight source snapshots are incomplete")
    for name in PREFLIGHT_SOURCES:
        snapshot = verify_file(root, preflight["source_snapshot"][name], "preflight source snapshot " + name)
        require(sha256(snapshot) == report["source_sha256"][name], "Preflight source snapshot differs: " + name)
    require(set(preflight.get("resource_snapshot", {})) == set(RESOURCE_NAMES),
            "Preflight Punkt resource snapshots are incomplete")
    for name in RESOURCE_NAMES:
        snapshot = verify_file(root, preflight["resource_snapshot"][name], "Punkt snapshot " + name)
        require(sha256(snapshot) == report["nltk_english_resource_sha256"][name], "Punkt snapshot differs: " + name)
    environment = plan.get("environment", {})
    frozen_packages = environment.get("package_versions", {})
    require(set(PREFLIGHT_PACKAGES) | {"torch"} <= set(frozen_packages), "Sealed environment metadata is incomplete")
    require(package_versions(frozen_packages) == frozen_packages, "Runtime package versions differ from sealed environment")
    require(platform.python_version() == environment.get("python_version"), "Runtime Python version differs from sealed environment")
    require(resource_hashes() == environment.get("nltk_english_resource_sha256"), "Runtime Punkt resources changed")
    sealed_at = parse_timestamp(plan.get("sealed_at_utc"), "sealed_at_utc")
    require(parse_timestamp(report["completed_at_utc"], "preflight completion") <= sealed_at <= datetime.now(timezone.utc),
            "Seal time must follow preflight and must not be in the future")
    return inputs, sources, environment


def load_freeze(root, cfg, input_paths, source_names):
    """Return stable JSON provenance after local upload/integrity checks.

    --check-only runners should skip this call and say the gate was not checked.
    Normal execution and resume must call it before loading any model. The same
    unchanged artifacts produce the same returned dictionary on every check.
    """
    root = Path(root).resolve()
    try:
        seal_file, upload_file = inside(root, SEAL_PATH), inside(root, UPLOAD_PATH)
        plan, uploaded = read_json(seal_file), read_json(upload_file)
        inputs, sources, environment = _verify_plan(root, cfg, input_paths, source_names, plan)
        require(uploaded.get("schema_version") == 1 and uploaded.get("status") == "uploaded",
                "Final inference requires the user's Moodle upload record")
        require(uploaded.get("evidence_basis") == "user_provided_moodle_receipt"
                and uploaded.get("user_attested_upload") is True,
                "The upload must be attested by the user with an actual receipt file")
        require(uploaded.get("sealed_plan_sha256") == sha256(seal_file), "Upload record does not bind the sealed plan")
        require(uploaded.get("protocol_sha256") == plan["protocol"]["sha256"], "Upload record does not bind the sealed Word document")
        require(uploaded.get("timestamp_precision_tolerance_seconds") == UPLOAD_PRECISION_TOLERANCE_SECONDS,
                "Unexpected Moodle timestamp precision policy")
        upload_time = parse_timestamp(uploaded.get("uploaded_at"), "Moodle uploaded_at")
        registered = parse_timestamp(uploaded.get("registered_at_utc"), "registered_at_utc")
        sealed_at = parse_timestamp(plan["sealed_at_utc"], "sealed_at_utc")
        require(upload_time >= sealed_at - timedelta(seconds=UPLOAD_PRECISION_TOLERANCE_SECONDS)
                and sealed_at <= registered <= datetime.now(timezone.utc)
                and upload_time <= registered,
                "Upload timestamp must follow sealing (60-second minute-resolution tolerance) and cannot be in the future")
        receipt = verify_file(root, uploaded.get("receipt"), "user-provided Moodle receipt")
        protected_hashes = {record["sha256"] for record in plan["inputs"].values()} | set(sources.values()) | {
            sha256(seal_file), plan["protocol"]["sha256"], plan["preflight"]["report"]["sha256"],
            plan["preflight"]["config_snapshot"]["sha256"]}
        require(sha256(receipt) not in protected_hashes, "The receipt cannot be a source, dataset, configuration or the protocol document itself")
        return {
            "status": "uploaded", "evidence_basis": "user_provided_moodle_receipt",
            "uploaded_at": uploaded["uploaded_at"],
            "sealed_plan_sha256": sha256(seal_file), "freeze_manifest_sha256": sha256(upload_file),
            "config_sha256": plan["inputs"]["config"]["sha256"], "protocol_sha256": plan["protocol"]["sha256"],
            "receipt_sha256": sha256(receipt), "preflight_report_sha256": plan["preflight"]["report"]["sha256"],
            "input_sha256": {name: plan["inputs"][name]["sha256"] for name in INPUT_KEYS},
            "source_sha256": sources, "package_versions": environment["package_versions"],
            "python_version": environment["python_version"],
            "nltk_english_resource_sha256": environment["nltk_english_resource_sha256"],
            "timestamp_precision_tolerance_seconds": UPLOAD_PRECISION_TOLERANCE_SECONDS,
            "verification_scope": "Local hashes, runtime metadata and user-provided evidence checked; Moodle server authenticity is not verified",
        }
    except FreezeError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise FreezeError("Freeze evidence is incomplete or invalid: " + str(exc)) from exc
