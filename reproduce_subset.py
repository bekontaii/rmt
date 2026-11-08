"""Fresh, explicitly operated 20-question/60-answer generation reproduction.

Copy this file into the Assignment 4 project root. Use the original pinned
environment (requirements.lock.txt and the sealed Python version), locally
cached pinned FLAN model, and both completed main runs. --check-only loads no
model and writes nothing. --execute never changes a main result or upload
receipt. An operator label is self-reported, not proof of classmate identity.

Example (replace paths and supply the actual operator):
  python reproduce_subset.py --retrieval-run final/retrieval_runs/RUN \
    --generation-run final/generation_runs/RUN --check-only
  python reproduce_subset.py --retrieval-run final/retrieval_runs/RUN \
    --generation-run final/generation_runs/RUN --operator "Actual operator" \
    --execute --out reproduction/UNIQUE_UTC_FOLDER
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
STRATEGIES = ["fixed", "overlap", "semantic"]
SELECTION_RULE = "First generation_question_id in frozen order for each article in frozen article order"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def choose_subset(article_titles, generation_ids, questions_by_id):
    """Select solely from the frozen ordered manifest, never from outcomes."""
    require(len(article_titles) == len(set(article_titles)) == 20,
            "Expected exactly 20 distinct frozen articles")
    first = {}
    for qid in generation_ids:
        require(qid in questions_by_id, "Unknown planned generation question")
        article = questions_by_id[qid]["article"]
        require(article in article_titles, "Unknown planned article")
        first.setdefault(article, qid)
    require(set(first) == set(article_titles), "An article has no planned generation question")
    chosen = [first[article] for article in article_titles]
    require(len(set(chosen)) == 20, "Reproduction question identities are not distinct")
    return chosen


def resolve_run(root, value, family):
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    require((root / "final" / family).resolve() in path.parents,
            "Use an explicit completed main " + family + " directory")
    return path


def verify(root, retrieval, generation):
    # Imports are intentionally delayed. These modules do not load models at import.
    import final_generation as main_gen
    from run_main_experiment import execution_proof

    cfg, paths, corpus, gen_ids, records, retrieval_manifest = main_gen.load_source(root, retrieval)
    proof = execution_proof(root, cfg, paths, main_gen.SOURCES)
    require(retrieval_manifest["freeze_manifest_sha256"] == proof["freeze_manifest_sha256"],
            "Retrieval provenance differs from the current verified local seal")
    manifest = read(generation / "run_manifest.json")
    require(manifest.get("purpose") == "final_generation" and manifest.get("status") == "completed"
            and manifest.get("final_run") is True and manifest.get("completed_records") == 3000
            and manifest.get("question_ids") == gen_ids, "Main generation must be completed (3000 answers)")
    identity = manifest["identity"]
    require(manifest["identity_sha256"] == main_gen.digest(identity), "Main generation identity checksum differs")
    require(identity["retrieval_source"] == str(retrieval.relative_to(root))
            and identity["retrieval_manifest_sha256"] == sha(retrieval / "run_manifest.json")
            and identity["retrieval_output_sha256"] == retrieval_manifest["output_sha256"]
            and identity["retrieval_source_sha256"] == retrieval_manifest["source_sha256"],
            "Main generation belongs to a different retrieval run")
    require(identity["freeze_proof"] == proof and identity["input_sha256"] == {n: sha(p) for n, p in paths.items()}
            and identity["config_sha256"] == sha(retrieval / "config_snapshot.json"),
            "Main generation seal/config/data proof differs")
    require(identity["source_sha256"] == {n: sha(root / n) for n in main_gen.SOURCES},
            "Current generation source differs from the main run")
    mandatory = {"summary.json", "per_question.jsonl", "config_snapshot.json", "corpus.json",
                 "questions.json", "selection_manifest.json", "retrieval_run_manifest.json"}
    mandatory.update(f"{s}/{name}" for s in STRATEGIES for name in ["per_question.json", "summary.json"])
    mandatory.update(f"{s}/answers/{qid}.json" for s in STRATEGIES for qid in gen_ids)
    require(mandatory <= set(manifest["output_sha256"]), "Completed main generation artifact roster is incomplete")
    for name, expected in manifest["output_sha256"].items():
        require(sha(main_gen.inside(generation, name)) == expected, "Main generation artifact changed: " + name)
    for name, expected in identity["source_sha256"].items():
        require(sha(generation / "source_snapshot" / name) == expected, "Main source snapshot changed: " + name)
    for filename, key in [("config_snapshot.json", "config"), ("corpus.json", "corpus"),
                          ("questions.json", "questions"), ("selection_manifest.json", "selection_manifest")]:
        require(sha(generation / filename) == identity["input_sha256"][key], "Main generation snapshot differs: " + filename)
    saved = main_gen.existing_answers(generation, records, gen_ids, manifest, cfg)
    require(len(saved) == 3000, "Main run is missing one or more validated answers")
    titles = retrieval_manifest["article_titles"]
    selected = choose_subset(titles, gen_ids, records["fixed"])
    current_environment = main_gen.versions()
    require(current_environment["packages"] == identity["environment"]["packages"]
            and current_environment["python"] == identity["environment"]["python"],
            "Use the original exact package versions and Python version; see requirements.lock.txt")
    stamp = dict(retrieval_manifest_sha256=sha(retrieval / "run_manifest.json"),
                 generation_manifest_sha256=sha(generation / "run_manifest.json"),
                 config_sha256=sha(paths["config"]), freeze_proof=proof,
                 source_sha256=identity["source_sha256"],
                 reproduction_script_sha256=sha(Path(__file__)))
    return main_gen, cfg, paths, corpus, records, saved, selected, stamp, current_environment


def comparisons(fresh, original):
    return dict(prediction_exact=fresh["prediction"] == original["prediction"],
                exact_match_equal=fresh["answer_metrics"]["exact_match"] == original["answer_metrics"]["exact_match"],
                f1_equal=abs(fresh["answer_metrics"]["f1"] - original["answer_metrics"]["f1"]) <= 1e-12,
                input_token_ids_equal=fresh["input_token_ids"] == original["input_token_ids"],
                attention_mask_equal=fresh["attention_mask"] == original["attention_mask"],
                output_token_ids_equal=fresh["output_token_ids"] == original["output_token_ids"],
                packed_prompt_equal=fresh["packed_prompt"] == original["packed_prompt"])


def execute(root, retrieval, generation, out, operator, verified):
    main_gen, cfg, paths, corpus, records, originals, chosen, stamp, environment = verified
    require(not out.exists(), "Choose a new reproduction output directory; existing runs are never overwritten")
    require((root / "reproduction").resolve() in out.parents, "Output must be a new folder under reproduction/")
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(purpose="generation_subset_reproduction", final_run=False, status="created",
                    started_at_utc=utc(), operator_label=operator,
                    operator_identity_basis="self_reported_label_not_verified_classmate_identity",
                    selection_rule=SELECTION_RULE, question_ids=chosen, n_questions=20, expected_records=60,
                    completed_records=0, retrieval_run=str(retrieval.relative_to(root)),
                    generation_run=str(generation.relative_to(root)), source_proof=stamp,
                    environment=environment, answer_sha256={},
                    scope="Fresh generation from saved main retrieval only; no retrieval rerun and no full-table reproduction claim",
                    moodle_upload_verified=stamp["freeze_proof"].get("moodle_upload_verified", False))
    main_gen.atomic(out / "run_manifest.json", manifest)
    started, active, results = time.perf_counter(), {"stage": "snapshot"}, []
    try:
        for name in [*main_gen.SOURCES, "run_main_experiment.py", "requirements.txt", "requirements.lock.txt"]:
            target = out / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        shutil.copyfile(Path(__file__), out / "source_snapshot" / "reproduce_subset.py")
        for name in ["config_snapshot.json", "selection_manifest.json"]:
            shutil.copyfile(generation / name, out / name)
        main_gen.atomic(out / "selected_questions.json", [records["fixed"][qid] for qid in chosen])
        active = {"stage": "model_loading"}
        manifest["status"] = "running"
        main_gen.atomic(out / "run_manifest.json", manifest)
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from run_development_generation import generate_record
        torch.manual_seed(42)
        gen = cfg["generation"]
        tick = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(gen["name"], revision=gen["revision"], use_fast=False,
                                                  legacy=True, local_files_only=True, trust_remote_code=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(gen["name"], revision=gen["revision"], dtype=torch.float32,
                                                      local_files_only=True, trust_remote_code=False).to("cpu").eval()
        require(next(model.parameters()).dtype == torch.float32 and next(model.parameters()).device.type == "cpu",
                "Expected the same pinned CPU float32 generator")
        manifest["model_loading_elapsed_s"] = time.perf_counter() - tick
        for strategy in STRATEGIES:
            for qid in chosen:
                active = dict(stage="answer_pipeline", strategy=strategy, qid=qid)
                tick, started_at = time.perf_counter(), utc()
                result = generate_record(records[strategy][qid], strategy, tokenizer, model, cfg, corpus, torch)
                elapsed = time.perf_counter() - tick
                match = comparisons(result, originals[(strategy, qid)])
                result.update(final_run=False, reproduction_run=True, operator_label=operator,
                              original_answer_sha256=sha(generation / strategy / "answers" / (qid + ".json")),
                              record_input_sha256=main_gen.digest(records[strategy][qid]),
                              timing=dict(started_at_utc=started_at, completed_at_utc=utc(), answer_pipeline_elapsed_s=elapsed),
                              comparison_with_main=match,
                              output_diagnostics=main_gen.output_diagnostics(result, model, tokenizer, gen["max_new_tokens"]))
                result["record_payload_sha256"] = main_gen.digest(result)
                answer = out / strategy / "answers" / (qid + ".json")
                main_gen.atomic(answer, result, create=True)
                results.append(result)
                manifest["completed_records"] = len(results)
                manifest["answer_sha256"][str(answer.relative_to(out))] = sha(answer)
                main_gen.atomic(out / "run_manifest.json", manifest)
                print(f"{strategy} {qid}: {len(results)}/60 fresh answers saved", flush=True)
        # Recheck immutable main evidence after fresh inference; never silently accept changed sources.
        require(verify(root, retrieval, generation)[7] == stamp, "Main inputs, source, results or provenance changed")
        keys = list(results[0]["comparison_with_main"])
        summary = dict(purpose="generation_subset_reproduction", n_questions=20, n_answers=60,
                       selection_rule=SELECTION_RULE, operator_label=operator,
                       operator_identity_verified=False, full_table_reproduction=False,
                       matches={k: sum(r["comparison_with_main"][k] for r in results) for k in keys},
                       match_denominator=60, f1_equality_absolute_tolerance=1e-12,
                       methods={s: dict(n_questions=20,
                                        matches={k: sum(r["comparison_with_main"][k] for r in results if r["strategy"] == s) for k in keys},
                                        fresh_exact_match=sum(r["answer_metrics"]["exact_match"] for r in results if r["strategy"] == s) / 20,
                                        fresh_f1=sum(r["answer_metrics"]["f1"] for r in results if r["strategy"] == s) / 20)
                                for s in STRATEGIES},
                       mismatch_cases=[dict(strategy=r["strategy"], id=r["id"], checks=r["comparison_with_main"])
                                       for r in results if not all(r["comparison_with_main"].values())],
                       interpretation="This is a predetermined subset generation check from saved retrieval. Report actual matches and mismatches; operator/peer identity requires separate evidence.")
        main_gen.atomic(out / "per_question.jsonl", results, jsonl=True)
        main_gen.atomic(out / "summary.json", summary)
        manifest.update(status="completed", completed_at_utc=utc(), elapsed_s=time.perf_counter()-started)
        manifest["output_sha256"] = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
                                     if p.is_file() and p.name != "run_manifest.json"}
        main_gen.atomic(out / "run_manifest.json", manifest)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        stopped_at_utc=utc(), elapsed_s=time.perf_counter()-started, active=active,
                        error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        main_gen.atomic(out / "run_manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrieval-run", required=True)
    parser.add_argument("--generation-run", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--operator")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    require((ROOT / "final_generation.py").is_file(), "Copy this script to the Assignment 4 project root before use")
    if args.execute:
        require(args.operator is not None and bool(args.operator.strip()), "Supply the actual operator name with --operator")
    retrieval = resolve_run(ROOT, args.retrieval_run, "retrieval_runs")
    generation = resolve_run(ROOT, args.generation_run, "generation_runs")
    verified = verify(ROOT, retrieval, generation)
    if args.check_only:
        print(json.dumps(dict(status="ready_for_explicit_subset_execution", models_loaded=False, files_written=False,
                              selection_rule=SELECTION_RULE, question_ids=verified[6], n_questions=20,
                              n_expected_answers=60, source_proof=verified[7],
                              environment=verified[8], peer_reproduction_performed=False), ensure_ascii=False, indent=2))
        return
    output = args.out or Path("reproduction") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (ROOT / output).resolve() if not output.is_absolute() else output.resolve()
    execute(ROOT, retrieval, generation, output, args.operator.strip(), verified)


if __name__ == "__main__":
    main()
