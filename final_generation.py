"""Final generation from one completed final retrieval run; safely resumable.

No model is loaded by import, --check-config, or --check-only. Actual execution
requires the independently recorded protocol/upload gate. No quality settings
or question identities are selected from the observed final outcomes.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import tempfile
import time
import traceback

STRATEGIES = ["fixed", "overlap", "semantic"]
SOURCES = ["final_generation.py", "run_development_generation.py", "prompt_builder.py",
           "metrics.py", "retrieval_metrics.py", "check_final_inputs.py", "chunkers.py",
           "semantic_chunker.py", "sentence_splitter.py", "freeze_gate.py"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            hasher.update(block)
    return hasher.hexdigest()


def inside(folder, name):
    path = (folder / name).resolve()
    require(folder.resolve() in path.parents, "Path escapes its declared folder")
    return path


def atomic(path, value, create=False, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not create or not path.exists(), "Refusing to overwrite an existing answer")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix="." + path.name + ".", suffix=".tmp", encoding="utf-8", delete=False) as stream:
            temporary = Path(stream.name)
            if jsonl:
                for record in value:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            else:
                json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def versions():
    packages = {}
    for name in ["numpy", "torch", "transformers", "tokenizers", "sentencepiece", "huggingface-hub", "safetensors", "sentence-transformers", "nltk"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return dict(packages=packages, python=platform.python_version(), platform=platform.platform(), machine=platform.machine())


def input_paths(root, cfg):
    data = cfg["data"]
    return dict(config=root / "experiment_config.json", raw_dataset=inside(root, data["dataset_path"]),
                selection_manifest=inside(root, data["selection_manifest_path"]),
                corpus=inside(root, data["corpus_path"]), questions=inside(root, data["questions_path"]))


def policy(cfg):
    gen = cfg["generation"]
    require(cfg["runtime"] == {"device": "cpu", "dtype": "float32"}, "Expected unchanged CPU float32 runtime")
    require(cfg["seed"] == 42 and cfg["chunking"]["strategies"] == STRATEGIES, "Seed/strategy policy changed")
    require(gen["name"] == "google/flan-t5-large" and gen["revision"] == "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
            and gen["context_top_k"] == 3 and gen["max_input_tokens"] == 512 and gen["max_new_tokens"] == 32
            and gen["do_sample"] is False and gen["num_beams"] == gen["batch_size"] == 1
            and gen["context_policy"] == "ranked_prefix_whole_words", "Generation policy differs from the tested development policy")


def load_source(root, source):
    """Validate saved results, not rerun retrieval or alter its ordered sample."""
    require((root / "final" / "retrieval_runs").resolve() in source.parents, "Expected an explicit final retrieval run")
    run = read(source / "run_manifest.json")
    require(run["purpose"] == "final_retrieval" and run["status"] == "completed"
            and run["completed_strategies"] == STRATEGIES and run["completed_records"] == 6000, "Source retrieval is incomplete or not final")
    mandatory = {"config_snapshot.json", "corpus.json", "questions.json", "selection_manifest.json", "query_embeddings.npy"}
    mandatory.update(f"{s}/{name}" for s in STRATEGIES for name in ["chunks.json", "chunk_embeddings.npy", "per_question.json", "summary.json"])
    require(mandatory <= set(run["output_sha256"]), "Retrieval artifact roster is incomplete")
    for name, expected in run["output_sha256"].items():
        require(sha(inside(source, name)) == expected, "Retrieval artifact changed: " + name)
    for name, expected in run["source_sha256"].items():
        require(sha(inside(source, "source_snapshot/" + name)) == expected, "Retrieval source snapshot changed")
    cfg = read(source / "config_snapshot.json")
    policy(cfg)
    paths = input_paths(root, cfg)
    require(all(sha(path) == run["input_sha256"][name] for name, path in paths.items()), "Current frozen inputs differ from retrieval inputs")
    for filename, key in [("config_snapshot.json", "config"), ("corpus.json", "corpus"), ("questions.json", "questions"), ("selection_manifest.json", "selection_manifest")]:
        require(sha(source / filename) == run["input_sha256"][key], "Input/snapshot mismatch")
    corpus, questions, selection = read(source / "corpus.json"), read(source / "questions.json"), read(source / "selection_manifest.json")
    ids, gen_ids, titles = [q["id"] for q in questions], selection["generation_question_ids"], selection["article_titles"]
    require(len(titles) == len(set(titles)) == 20 and list(corpus) == titles == run["article_titles"], "Expected the same 20 source articles")
    require(len(ids) == len(set(ids)) == 2000 and ids == selection["retrieval_question_ids"] == run["question_ids"], "Retrieval ID order/count mismatch")
    require(len(gen_ids) == len(set(gen_ids)) == 1000 and gen_ids == run["generation_question_ids"]
            and set(gen_ids) <= set(ids), "Generation subset identities/order changed")
    require(all(isinstance(qid, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", qid) for qid in ids), "Unsafe question ID")
    by_id = {q["id"]: q for q in questions}
    require(Counter(q["article"] for q in questions) == Counter({title: 100 for title in titles})
            and Counter(by_id[qid]["article"] for qid in gen_ids) == Counter({title: 50 for title in titles}), "Unbalanced article question counts")
    require(all(q["for_generation"] is (q["id"] in set(gen_ids)) for q in questions), "Generation flags differ from the frozen subset")
    records = {}
    for strategy in STRATEGIES:
        rows, chunks = read(source / strategy / "per_question.json"), read(source / strategy / "chunks.json")
        require([row["id"] for row in rows] == ids, "Strategies have different retrieval question identities/order")
        by_chunk = {c["chunk_id"]: c for c in chunks}
        require(len(by_chunk) == len(chunks), "Duplicate source chunk IDs")
        for row, q in zip(rows, questions):
            require(all(row[key] == value for key, value in q.items()), "Saved question content differs")
            ranked = row["retrieved_results"]
            require(len(ranked) == 5 and [r["rank"] for r in ranked] == [1, 2, 3, 4, 5]
                    and len({r["chunk"]["chunk_id"] for r in ranked}) == 5, "Malformed top-five retrieval")
            for result in ranked:
                chunk = result["chunk"]
                require(chunk == by_chunk[chunk["chunk_id"]] and corpus[chunk["article"]][chunk["start"]:chunk["end"]] == chunk["text"]
                        and math.isfinite(result["score"]) and abs(result["score"]) <= 1.0001, "Invalid retrieved source span/score")
            for k in [1, 3, 5]:
                hit = float(any(c["chunk"]["article"] == row["article"] and c["chunk"]["start"] <= start < end <= c["chunk"]["end"]
                                for c in ranked[:k] for start, end in row["answer_spans"]))
                require(row["retrieval_metrics"][f"hit_at_{k}"] == hit, "Retrieval Hit differs from source offsets")
        records[strategy] = {row["id"]: row for row in rows if row["id"] in set(gen_ids)}
    return cfg, paths, corpus, gen_ids, records, run


def identity(root, source, cfg, paths, run, proof):
    return dict(config_sha256=sha(source / "config_snapshot.json"), input_sha256={n: sha(p) for n, p in paths.items()},
                retrieval_source=str(source.relative_to(root)), retrieval_manifest_sha256=sha(source / "run_manifest.json"),
                retrieval_output_sha256=run["output_sha256"], retrieval_source_sha256=run["source_sha256"],
                source_sha256={n: sha(root / n) for n in SOURCES}, environment=versions(), freeze_proof=proof)


def validate_answer(record, row, strategy, identity_hash, gen):
    require(record["strategy"] == strategy and record["run_identity_sha256"] == identity_hash
            and record["record_input_sha256"] == digest(row) and record["final_run"] is True,
            "Existing answer belongs to different inputs/configuration/run")
    require(all(record[key] == value for key, value in row.items()), "Existing answer source fields changed")
    require(record["record_payload_sha256"] == digest({k: v for k, v in record.items() if k != "record_payload_sha256"}), "Answer payload checksum mismatch")
    packed = record["packed_prompt"]
    require(isinstance(record["prediction"], str) and packed["prompt"] == gen["prompt_template"].format(question=row["question"], context=packed["context"])
            and len(record["input_token_ids"]) == len(record["attention_mask"]) == packed["input_tokens"] <= 512
            and all(type(i) is int for i in record["input_token_ids"] + record["output_token_ids"])
            and all(i == 1 for i in record["attention_mask"]), "Malformed saved prompt/token trace")
    require(record["answer_metrics"]["exact_match"] in [0, 1] and 0 <= record["answer_metrics"]["f1"] <= 1
            and math.isfinite(record["answer_metrics"]["f1"]) and record["visible_evidence_hit"] in [0, 1]
            and record["visible_evidence_hit"] <= row["retrieval_metrics"]["hit_at_3"], "Invalid saved answer metrics")
    require(math.isfinite(record["timing"]["answer_pipeline_elapsed_s"]) and record["timing"]["answer_pipeline_elapsed_s"] >= 0,
            "Malformed saved runtime")


def existing_answers(out, records, gen_ids, manifest, cfg):
    saved = {}
    for path in out.glob("*/answers/*.json"):
        strategy, qid = path.parent.parent.name, path.stem
        require(strategy in STRATEGIES and qid in records[strategy] and qid in gen_ids, "Unexpected saved answer ID")
        record = read(path)
        validate_answer(record, records[strategy][qid], strategy, manifest["identity_sha256"], cfg["generation"])
        relative = str(path.relative_to(out))
        previous = manifest["answer_sha256"].get(relative)
        require(previous is None or previous == sha(path), "An existing checkpoint answer was modified")
        saved[(strategy, qid)] = record
    require(set(manifest["answer_sha256"]) <= {f"{s}/answers/{q}.json" for s, q in saved}, "A committed answer file is missing")
    return saved


def output_diagnostics(result, model, tokenizer, max_tokens):
    start = model.generation_config.decoder_start_token_id
    ids = result["output_token_ids"]
    generated = ids[1:] if ids and ids[0] == start else ids
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos if eos is not None else tokenizer.eos_token_id])
    ended = any(token in eos for token in generated)
    return dict(generated_token_count=len(generated), count_includes_eos=True, output_has_eos=ended,
                output_reached_max_new_tokens=len(generated) >= max_tokens,
                output_may_be_length_limited=len(generated) >= max_tokens and not ended,
                max_new_tokens=max_tokens, decoder_start_token_id=start, eos_token_ids=sorted(x for x in eos if x is not None))


def summarize(rows):
    require(len(rows) == 1000, "Each strategy must have exactly 1000 generated answers")
    return dict(n_questions=1000, exact_match=sum(r["answer_metrics"]["exact_match"] for r in rows)/1000,
                f1=sum(r["answer_metrics"]["f1"] for r in rows)/1000,
                hit_at_3=sum(r["retrieval_metrics"]["hit_at_3"] for r in rows)/1000,
                visible_evidence_hit=sum(r["visible_evidence_hit"] for r in rows)/1000,
                context_truncated_count=sum(r["packed_prompt"]["context_truncated"] for r in rows),
                output_may_be_length_limited_count=sum(r["output_diagnostics"]["output_may_be_length_limited"] for r in rows),
                mean_generated_tokens=sum(r["output_diagnostics"]["generated_token_count"] for r in rows)/1000,
                answer_pipeline_total_s=sum(r["timing"]["answer_pipeline_elapsed_s"] for r in rows),
                answer_pipeline_mean_s=sum(r["timing"]["answer_pipeline_elapsed_s"] for r in rows)/1000,
                timing_scope="Prompt packing, tokenization, generation, decoding and scoring; observed CPU time, not pure model latency or a dedicated benchmark")


def execute(root, source, out, cfg, paths, corpus, gen_ids, records, run, proof, resume):
    current = identity(root, source, cfg, paths, run, proof)
    if resume:
        manifest = read(out / "run_manifest.json")
        require(manifest["purpose"] == "final_generation" and manifest["identity"] == current
                and manifest["identity_sha256"] == digest(current) and manifest["question_ids"] == gen_ids,
                "Resume rejected: immutable inputs, source, packages, parameters or gate proof changed")
    else:
        out.mkdir(parents=True, exist_ok=False)
        manifest = dict(purpose="final_generation", final_run=True, status="created", run_id=out.name,
                        started_at_utc=utc(), expected_records=3000, completed_records=0, question_ids=gen_ids,
                        retrieval_question_ids=run["question_ids"], article_titles=run["article_titles"],
                        identity=current, identity_sha256=digest(current), attempts=[], answer_sha256={})
        for filename in ["config_snapshot.json", "corpus.json", "questions.json", "selection_manifest.json"]:
            shutil.copyfile(source / filename, out / filename)
        shutil.copyfile(source / "run_manifest.json", out / "retrieval_run_manifest.json")
        for name in SOURCES:
            target = out / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        atomic(out / "run_manifest.json", manifest)
    with (out / "generation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read(out / "run_manifest.json")
        require(manifest["identity"] == current and manifest["identity_sha256"] == digest(current), "Run identity changed before writer lock")
        saved = existing_answers(out, records, gen_ids, manifest, cfg)
        for filename in ["config_snapshot.json", "corpus.json", "questions.json", "selection_manifest.json"]:
            require(sha(out / filename) == sha(source / filename), "Generation input snapshot changed")
        require(all(sha(out / "source_snapshot" / n) == h for n, h in current["source_sha256"].items()), "Generation code snapshot changed")
        if manifest["status"] == "completed":
            require(len(saved) == 3000 and all(sha(inside(out, n)) == h for n, h in manifest["output_sha256"].items()), "Completed artifacts changed")
            print("Completed run verified; no answers regenerated.")
            return
        attempt = dict(number=len(manifest["attempts"])+1, status="running", started_at_utc=utc(), resumed=resume, existing_answers=len(saved))
        manifest["attempts"].append(attempt)
        manifest.update(status="running", completed_records=len(saved))
        for strategy, qid in saved:
            name = f"{strategy}/answers/{qid}.json"
            manifest["answer_sha256"][name] = sha(out / name)
        atomic(out / "run_manifest.json", manifest)
        started, active = time.perf_counter(), {"stage": "model_loading"}
        old_term = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
        try:
            os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            from run_development_generation import generate_record
            torch.manual_seed(cfg["seed"])
            gen = cfg["generation"]
            tokenizer = AutoTokenizer.from_pretrained(gen["name"], revision=gen["revision"], use_fast=False, legacy=True, local_files_only=True, trust_remote_code=False)
            model = AutoModelForSeq2SeqLM.from_pretrained(gen["name"], revision=gen["revision"], dtype=torch.float32, local_files_only=True, trust_remote_code=False).to("cpu").eval()
            require(next(model.parameters()).dtype == torch.float32 and next(model.parameters()).device.type == "cpu", "Generator device/dtype mismatch")
            attempt["model_loading_elapsed_s"] = time.perf_counter()-started
            for strategy in STRATEGIES:
                for qid in gen_ids:
                    if (strategy, qid) in saved:
                        continue
                    active = dict(stage="answer_pipeline", strategy=strategy, qid=qid)
                    start_utc, tick = utc(), time.perf_counter()
                    result = generate_record(records[strategy][qid], strategy, tokenizer, model, cfg, corpus, torch)
                    stop, stop_utc = time.perf_counter(), utc()
                    result.update(final_run=True, run_identity_sha256=manifest["identity_sha256"],
                                  record_input_sha256=digest(records[strategy][qid]), attempt=attempt["number"],
                                  timing=dict(started_at_utc=start_utc, completed_at_utc=stop_utc, perf_counter_start=tick,
                                              perf_counter_end=stop, answer_pipeline_elapsed_s=stop-tick),
                                  output_diagnostics=output_diagnostics(result, model, tokenizer, gen["max_new_tokens"]))
                    result["record_payload_sha256"] = digest(result)
                    validate_answer(result, records[strategy][qid], strategy, manifest["identity_sha256"], gen)
                    answer_path = out / strategy / "answers" / (qid + ".json")
                    atomic(answer_path, result, create=True)
                    saved[(strategy, qid)] = result
                    manifest["answer_sha256"][str(answer_path.relative_to(out))] = sha(answer_path)
                    manifest["completed_records"] = len(saved)
                    atomic(out / "run_manifest.json", manifest)
                    print(f"{strategy} {qid}: {len(saved)}/3000 saved", flush=True)
            saved = existing_answers(out, records, gen_ids, manifest, cfg)
            require(len(saved) == 3000 and identity(root, source, cfg, paths, run, proof) == current
                    and all(sha(inside(source, n)) == h for n, h in run["output_sha256"].items()), "Completion count/input provenance changed")
            from freeze_gate import load_freeze
            require(load_freeze(root, cfg, paths, SOURCES) == proof, "Freeze/upload proof changed during generation")
            summaries, canonical = {}, []
            for strategy in STRATEGIES:
                rows = [saved[(strategy, qid)] for qid in gen_ids]
                summaries[strategy] = summarize(rows)
                atomic(out / strategy / "per_question.json", rows)
                atomic(out / strategy / "summary.json", summaries[strategy])
                canonical.extend(rows)
            atomic(out / "per_question.jsonl", canonical, jsonl=True)
            attempt.update(status="completed", completed_at_utc=utc(), elapsed_s=time.perf_counter()-started)
            atomic(out / "summary.json", dict(purpose="final_generation", n_questions=1000, completed_records=3000,
                   methods=summaries, attempts=manifest["attempts"], total_attempt_elapsed_s=sum(a.get("elapsed_s", 0) for a in manifest["attempts"])))
            manifest.update(status="completed", completed_at_utc=utc(), output_sha256={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
                            if p.is_file() and p.name not in {"run_manifest.json", "generation.lock"} and not p.name.endswith(".tmp")})
            atomic(out / "run_manifest.json", manifest)
        except BaseException as error:
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            attempt.update(status=status, stopped_at_utc=utc(), elapsed_s=time.perf_counter()-started,
                           active=active, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            manifest.update(status=status, completed_records=len(saved))
            atomic(out / "run_manifest.json", manifest)
            raise
        finally:
            signal.signal(signal.SIGTERM, old_term)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--retrieval-run", type=Path)
    modes.add_argument("--resume", type=Path)
    modes.add_argument("--check-config", action="store_true", help="Validate final metadata/config without receipt, source run or model")
    parser.add_argument("--check-only", action="store_true", help="Validate the supplied saved run; skip gate and model loading")
    args = parser.parse_args()
    if args.check_config:
        from check_final_inputs import validate_inputs
        cfg = read(root / "experiment_config.json")
        policy(cfg)
        paths = input_paths(root, cfg)
        _, questions, titles, _ = validate_inputs(cfg, paths)
        selected = read(paths["selection_manifest"])["generation_question_ids"]
        print(json.dumps(dict(status="metadata_validated", config_status=cfg.get("status"), articles=len(titles),
              retrieval_questions=len(questions), generation_ids_sha256=digest(selected), generation_questions=len(selected),
              expected_answers=3000, gate_checked=False, models_loaded=False), indent=2))
        return
    out = (root / args.resume).resolve() if args.resume else None
    if out is not None:
        require((root / "final" / "generation_runs").resolve() in out.parents, "Resume path is not a final generation run")
        previous = read(out / "run_manifest.json")
        source = inside(root, previous["identity"]["retrieval_source"])
    else:
        source = (root / args.retrieval_run).resolve()
    cfg, paths, corpus, gen_ids, records, run = load_source(root, source)
    if args.check_only:
        if out is not None:
            require(previous["purpose"] == "final_generation" and previous["identity_sha256"] == digest(previous["identity"]), "Malformed generation identity")
            require(identity(root, source, cfg, paths, run, previous["identity"]["freeze_proof"]) == previous["identity"], "Saved immutable metadata changed")
            count = len(existing_answers(out, records, gen_ids, previous, cfg))
        else:
            count = 0
        print(json.dumps(dict(status="saved_metadata_validated", articles=20, generation_questions=1000,
              expected_answers=3000, saved_answers=count, gate_checked=False, models_loaded=False, final_execution=False), indent=2))
        return
    from freeze_gate import load_freeze
    proof = load_freeze(root, cfg, paths, SOURCES)
    require(run["freeze_manifest_sha256"] == proof["freeze_manifest_sha256"], "Retrieval and generation use different freezes")
    if out is None:
        out = root / "final" / "generation_runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    execute(root, source, out, cfg, paths, corpus, gen_ids, records, run, proof, args.resume is not None)


if __name__ == "__main__":
    main()
