import argparse
import hashlib
import json
import math
import platform
import shutil
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from metrics import score_answer
from prompt_builder import build_prompt
from retrieval_metrics import answer_span_hit_at_k


ARTICLES = ["Warsaw", "Martin_Luther", "Teacher"]
STRATEGIES = ["fixed", "overlap", "semantic"]
GENERATOR = "google/flan-t5-large"
REVISION = "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(folder, name):
    path = (folder / name).resolve()
    require(folder.resolve() in path.parents, "A saved file path escapes its folder")
    return path


def check_chunk(chunk, corpus):
    article, start, end = chunk["article"], chunk["start"], chunk["end"]
    require(article in corpus and type(start) is int and type(end) is int
            and 0 <= start < end <= len(corpus[article])
            and corpus[article][start:end] == chunk["text"], "Chunk/source mismatch")


def load_retrieval(root, source):
    base = (root / "development" / "retrieval_runs").resolve()
    require(base in source.parents and source.is_dir(), "Select a development retrieval run")
    run = read_json(source / "run_manifest.json")
    require(run["purpose"] == "development_retrieval" and run["status"] == "completed"
            and run["article_titles"] == ARTICLES and run["completed_strategies"] == STRATEGIES,
            "Expected a completed three-article development retrieval run")
    hashes = run["output_sha256"]
    required = {"config_snapshot.json", "corpus.json", "questions.json", "selection_manifest.json"}
    required.update(f"{s}/{name}" for s in STRATEGIES
                    for name in ["chunks.json", "per_question.json", "summary.json"])
    require(required <= set(hashes), "Retrieval manifest omits required artifacts")
    for name, expected in hashes.items():
        require(sha256(inside(source, name)) == expected, f"Retrieval artifact changed: {name}")
    for name, expected in run["source_sha256"].items():
        require(sha256(inside(source, "source_snapshot/" + name)) == expected, "Retrieval source snapshot mismatch")
    cfg = read_json(source / "config_snapshot.json")
    corpus, questions = read_json(source / "corpus.json"), read_json(source / "questions.json")
    dev = read_json(source / "selection_manifest.json")
    for filename, key in [("config_snapshot.json", "config"), ("corpus.json", "corpus"),
                          ("questions.json", "questions"), ("selection_manifest.json", "development_manifest")]:
        require(hashes[filename] == run["input_sha256"][key], "Snapshot/input hash mismatch")
    final_path = inside(root, cfg["data"]["selection_manifest_path"])
    require(sha256(final_path) == run["input_sha256"]["final_manifest"]
            == dev["final_selection_manifest_sha256"], "Final selection metadata changed")
    final = read_json(final_path)
    raw_path = inside(root, cfg["data"]["dataset_path"])
    require(sha256(raw_path) == run["input_sha256"]["raw_dataset"]
            == cfg["data"]["dataset_sha256"] == dev["dataset_sha256"]
            == final["dataset_sha256"], "Raw dataset hash mismatch")
    require(dev["purpose"] == "development_pilot" and list(corpus) == ARTICLES
            and dev["article_titles"] == ARTICLES, "Not the prepared development sample")
    require(cfg["seed"] == run["seed"] == dev["seed"] == 42, "Development seed mismatch")
    ids = [q["id"] for q in questions]
    require(len(ids) == len(set(ids)) == 9 and ids == run["question_ids"]
            == dev["retrieval_question_ids"] == dev["generation_question_ids"], "Question IDs/order mismatch")
    require(Counter(q["article"] for q in questions) == Counter({a: 3 for a in ARTICLES}),
            "Expected three questions per article")
    require(set(ARTICLES) <= set(final["excluded_a3_articles"])
            and not set(ARTICLES) & set(final["article_titles"])
            and not set(ids) & set(final["retrieval_question_ids"]), "Development/final overlap")
    for q in questions:
        require(q["for_generation"] is True and bool(q["answers"])
                and len(q["answers"]) == len(q["answer_spans"]), "Invalid generation question")
        for answer, (start, end) in zip(q["answers"], q["answer_spans"]):
            check_chunk(dict(article=q["article"], start=start, end=end, text=answer), corpus)
    gen = cfg["generation"]
    require(cfg["runtime"]["device"] == "cpu" and cfg["runtime"]["dtype"] == "float32",
            "This runner supports CPU float32 only")
    require(cfg["chunking"]["strategies"] == STRATEGIES and cfg["retrieval"]["top_k"] == 5,
            "Unexpected retrieval configuration")
    require(gen["name"] == GENERATOR and gen["revision"] == REVISION
            and gen["context_top_k"] == 3 and gen["max_input_tokens"] == 512
            and gen["max_new_tokens"] == 32 and gen["do_sample"] is False
            and gen["num_beams"] == 1 and gen["batch_size"] == 1
            and gen["context_policy"] == "ranked_prefix_whole_words", "Unsupported generation settings")
    require(sha256(root / "retrieval_metrics.py") == run["source_sha256"]["retrieval_metrics.py"],
            "Retrieval scorer changed since the source run")
    records = {}
    for strategy in STRATEGIES:
        rows = read_json(source / strategy / "per_question.json")
        chunks = read_json(source / strategy / "chunks.json")
        by_id = {c["chunk_id"]: c for c in chunks}
        require(len(by_id) == len(chunks), "Duplicate chunk IDs")
        for chunk in chunks:
            check_chunk(chunk, corpus)
        require([r["id"] for r in rows] == ids, "Strategies do not use identical question order")
        for q, row in zip(questions, rows):
            require(all(row[key] == value for key, value in q.items()), "Question content mismatch")
            ranked = row["retrieved_results"]
            require(len(ranked) == 5 and [r["rank"] for r in ranked] == [1, 2, 3, 4, 5], "Expected ranked top5")
            require(len({r["chunk"]["chunk_id"] for r in ranked}) == 5, "Duplicate retrieved chunks")
            for result in ranked:
                require(result["chunk"] == by_id[result["chunk"]["chunk_id"]]
                        and math.isfinite(result["score"]) and abs(result["score"]) <= 1.0001,
                        "Invalid retrieved chunk/score")
            require(all(a["score"] >= b["score"] for a, b in zip(ranked, ranked[1:])), "Unsorted retrieval")
            for k in [1, 3, 5]:
                require(row["retrieval_metrics"][f"hit_at_{k}"] == answer_span_hit_at_k(
                    ranked, row["article"], row["answer_spans"], k=k), "Saved retrieval Hit mismatch")
        records[strategy] = rows
    return cfg, corpus, questions, run, records


def package_versions():
    result = {}
    for name in ["torch", "transformers", "tokenizers", "sentencepiece", "huggingface-hub"]:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def generate_record(row, strategy, tokenizer, model, cfg, corpus, torch):
    gen = cfg["generation"]
    packed = build_prompt(row["question"], row["retrieved_results"], tokenizer, cfg)
    require(packed["prompt"] == gen["prompt_template"].format(
        question=row["question"], context=packed["context"]), "Question/template changed")
    full_context = "\n\n".join(r["chunk"]["text"] for r in row["retrieved_results"][:gen["context_top_k"]])
    require(full_context.startswith(packed["context"]), "Context is not an exact prefix")
    for visible in packed["visible_chunks"]:
        check_chunk(visible, corpus)
        original = row["retrieved_results"][visible["rank"] - 1]["chunk"]
        require(visible["start"] == original["start"] and visible["end"] <= original["end"]
                and visible["original_end"] == original["end"], "Visible chunk mapping mismatch")
    inputs = tokenizer(packed["prompt"], return_tensors="pt", add_special_tokens=True,
                       truncation=False, max_length=gen["max_input_tokens"]).to(cfg["runtime"]["device"])
    require(int(inputs["input_ids"].shape[1]) == packed["input_tokens"] <= gen["max_input_tokens"],
            "Actual input token count differs from the packed prompt")
    with torch.inference_mode():
        outputs = model.generate(**inputs, do_sample=gen["do_sample"], num_beams=gen["num_beams"],
                                 max_new_tokens=gen["max_new_tokens"])
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    visible_hit = float(any(c["article"] == row["article"] and c["start"] <= start < end <= c["end"]
                            for c in packed["visible_chunks"] for start, end in row["answer_spans"]))
    require(visible_hit <= row["retrieval_metrics"]["hit_at_3"], "Visible evidence exceeds retrieved top3 Hit")
    return dict(row, strategy=strategy, packed_prompt=packed, prediction=prediction,
                answer_metrics=score_answer(prediction, row["answers"]), visible_evidence_hit=visible_hit,
                input_token_ids=inputs["input_ids"][0].tolist(),
                attention_mask=inputs["attention_mask"][0].tolist(), output_token_ids=outputs[0].tolist())


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate only from a saved nine-question DEVELOPMENT retrieval run.")
    parser.add_argument("--retrieval-run", type=Path, required=True,
                        help="Explicit path to a completed development/retrieval_runs/<UTC> folder")
    args = parser.parse_args()
    source = (root / args.retrieval_run).resolve()
    cfg, corpus, questions, retrieval_run, records = load_retrieval(root, source)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = root / "development" / "generation_runs" / run_id
    out.mkdir(parents=True, exist_ok=False)
    names = [Path(__file__).name, "prompt_builder.py", "metrics.py", "retrieval_metrics.py"]
    manifest = {"purpose": "development_generation", "status": "running", "run_id": run_id,
                "started_at_utc": datetime.now(timezone.utc).isoformat(), "seed": cfg["seed"],
                "article_titles": ARTICLES, "question_ids": [q["id"] for q in questions],
                "retrieval_source": str(source.relative_to(root)),
                "retrieval_manifest_sha256": sha256(source / "run_manifest.json"),
                "config_sha256": sha256(source / "config_snapshot.json"),
                "retrieval_input_sha256": retrieval_run["input_sha256"],
                "retrieval_artifact_sha256": retrieval_run["output_sha256"],
                "source_sha256": {name: sha256(root / name) for name in names},
                "python_version": platform.python_version(), "platform": platform.platform(),
                "package_versions": package_versions(), "completed_records": 0}
    write_json(out / "run_manifest.json", manifest)
    try:
        for name in ["config_snapshot.json", "questions.json", "corpus.json", "selection_manifest.json"]:
            shutil.copyfile(source / name, out / name)
        shutil.copyfile(source / "run_manifest.json", out / "retrieval_run_manifest.json")
        (out / "source_snapshot").mkdir()
        for name in names:
            shutil.copyfile(root / name, out / "source_snapshot" / name)
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        torch.manual_seed(cfg["seed"])
        gen = cfg["generation"]
        print("Loading development generator:", gen["name"], flush=True)
        tokenizer = AutoTokenizer.from_pretrained(gen["name"], revision=gen["revision"],
                                                 use_fast=False, legacy=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(gen["name"], revision=gen["revision"],
                                                     dtype=torch.float32).to(cfg["runtime"]["device"]).eval()
        require(next(model.parameters()).dtype == torch.float32, "Generator is not float32")
        manifest["model"] = dict(name=gen["name"], revision=gen["revision"], device="cpu", dtype="float32",
                                 tokenizer_use_fast=False, tokenizer_legacy=True)
        summaries = {}
        with (out / "per_question.jsonl").open("x", encoding="utf-8") as stream:
            for strategy in STRATEGIES:
                folder = out / strategy
                folder.mkdir()
                generated = []
                for index, row in enumerate(records[strategy], start=1):
                    print(f"{strategy}: {index}/9 | {row['id']}", flush=True)
                    result = generate_record(row, strategy, tokenizer, model, cfg, corpus, torch)
                    stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    generated.append(result)
                    manifest["completed_records"] += 1
                    write_json(out / "run_manifest.json", manifest)
                summary = {"n_questions": len(generated),
                           "exact_match": sum(r["answer_metrics"]["exact_match"] for r in generated) / len(generated),
                           "f1": sum(r["answer_metrics"]["f1"] for r in generated) / len(generated),
                           "hit_at_3": sum(r["retrieval_metrics"]["hit_at_3"] for r in generated) / len(generated),
                           "visible_evidence_hit": sum(r["visible_evidence_hit"] for r in generated) / len(generated),
                           "context_truncated_count": sum(r["packed_prompt"]["context_truncated"] for r in generated)}
                write_json(folder / "per_question.json", generated)
                write_json(folder / "summary.json", summary)
                summaries[strategy] = summary
        require(manifest["completed_records"] == 27, "Expected exactly 27 development answers")
        require(sha256(source / "run_manifest.json") == manifest["retrieval_manifest_sha256"]
                and all(sha256(inside(source, n)) == h for n, h in retrieval_run["output_sha256"].items()),
                "Retrieval source changed during generation")
        require(all(sha256(root / n) == h for n, h in manifest["source_sha256"].items()), "Source code changed")
        write_json(out / "summary.json", summaries)
        manifest.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        output_sha256={str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*"))
                                       if p.is_file() and p.name != "run_manifest.json"})
        write_json(out / "run_manifest.json", manifest)
        print(json.dumps(summaries, indent=2))
        print("Saved DEVELOPMENT generation:", out)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}",
                        stopped_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(out / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
