import argparse
import hashlib
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from check_final_inputs import validate_inputs
from chunkers import fixed_chunks
from retrieval_metrics import answer_span_hit_at_k
from run_development_retrieval import encode_checked
from semantic_chunker import semantic_chunks


STRATEGIES = ["fixed", "overlap", "semantic"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(root, name):
    path = (root / name).resolve()
    require(root.resolve() in path.parents, "A project input path escapes this project")
    return path


def input_paths(root, cfg):
    return dict(config=root / "experiment_config.json",
                raw_dataset=inside(root, cfg["data"]["dataset_path"]),
                selection_manifest=inside(root, cfg["data"]["selection_manifest_path"]),
                corpus=inside(root, cfg["data"]["corpus_path"]),
                questions=inside(root, cfg["data"]["questions_path"]))


def validate_runtime(cfg):
    require(cfg["runtime"]["device"] == "cpu" and cfg["runtime"]["dtype"] == "float32",
            "This frozen runner supports CPU float32 only")
    require(cfg["embedding"]["batch_size"] == 32,
            "Batch32 is required to match the existing semantic unit encoder")
    retrieval = cfg["retrieval"]
    require(cfg["chunking"]["strategies"] == STRATEGIES
            and retrieval["top_k"] == 5 and retrieval["report_hit_at"] == [1, 3, 5]
            and retrieval["search_scope"] == "all_selected_articles"
            and retrieval["tie_break"] == "original_chunk_order", "Unsupported retrieval policy")


def build_final_chunks(strategy, corpus, article_order, model, sentence_tokenizer, cfg):
    setting, chunks = cfg["chunking"], []
    for number, article in enumerate(article_order, start=1):
        text = corpus[article]
        print(f"Building {strategy}: {number}/{len(article_order)} {article}", flush=True)
        if strategy == "semantic":
            pieces = semantic_chunks(article, text, model, sentence_tokenizer,
                                     max_words=setting["max_words"],
                                     breakpoint_percentile=setting["semantic_percentile"])
        else:
            overlap = setting["overlap_words"] if strategy == "overlap" else 0
            pieces = fixed_chunks(article, text, size=setting["max_words"], overlap=overlap)
        for piece in pieces:
            start, end = piece["start"], piece["end"]
            require(piece["article"] == article and type(start) is int and type(end) is int
                    and 0 <= start < end <= len(text) and piece["text"] == text[start:end],
                    "Chunk/source offsets mismatch")
            require(1 <= len(piece["text"].split()) <= setting["max_words"], "Chunk word cap violated")
            chunks.append(dict(piece, chunk_id=f"{strategy}:{len(chunks):05d}"))
    require(len(chunks) >= cfg["retrieval"]["top_k"], "Too few chunks for top5")
    return chunks


def summarize(records, cfg):
    def subset(rows):
        require(bool(rows), "Cannot summarize an empty question subset")
        result = {"n_questions": len(rows)}
        for k in cfg["retrieval"]["report_hit_at"]:
            name = f"hit_at_{k}"
            hits = sum(row["retrieval_metrics"][name] for row in rows)
            result[name] = hits / len(rows)
            result[name + "_count"] = int(hits)
        return result
    return dict(all_retrieval=subset(records),
                generation_subset=subset([row for row in records if row["for_generation"]]))


def package_versions():
    result = {}
    for name in ["numpy", "torch", "transformers", "sentence-transformers", "nltk",
                 "tokenizers", "huggingface-hub"]:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def nltk_resource_hashes():
    import nltk
    location = nltk.data.find("tokenizers/punkt_tab/english/")
    result = {}
    for name in ["abbrev_types.txt", "collocations.tab", "ortho_context.tab", "sent_starters.txt"]:
        with location.join(name).open() as stream:
            result[name] = hashlib.sha256(stream.read()).hexdigest()
    return result


def assert_unchanged(root, paths, manifest, source_names, cfg, load_freeze):
    require(all(sha256(path) == manifest["input_sha256"][name] for name, path in paths.items()),
            "An input changed during final retrieval")
    require(all(sha256(root / name) == value for name, value in manifest["source_sha256"].items()),
            "A source file changed during final retrieval")
    require(nltk_resource_hashes() == manifest["nltk_english_resource_sha256"], "NLTK resources changed")
    require(sha256(root / "protocol" / "freeze_manifest.json") == manifest["freeze_manifest_sha256"],
            "Freeze evidence changed during final retrieval")
    load_freeze(root, cfg, paths, source_names)


def execute(root, cfg, paths, corpus, questions, article_order, selection, source_names, proof, load_freeze):
    started = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = root / "final" / "retrieval_runs" / run_id
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(purpose="final_retrieval", status="running", run_id=run_id,
                    started_at_utc=datetime.now(timezone.utc).isoformat(), seed=cfg["seed"],
                    article_titles=article_order, question_ids=[q["id"] for q in questions],
                    generation_question_ids=selection["generation_question_ids"],
                    input_sha256={name: sha256(path) for name, path in paths.items()},
                    source_sha256={name: sha256(root / name) for name in source_names},
                    freeze_proof=proof, freeze_manifest_sha256=sha256(root / "protocol" / "freeze_manifest.json"),
                    python_version=platform.python_version(), platform=platform.platform(),
                    package_versions=package_versions(), completed_strategies=[], timing={}, completed_records=0)
    write_json(out / "run_manifest.json", manifest)
    try:
        for key, name in [("config", "config_snapshot.json"), ("corpus", "corpus.json"),
                          ("questions", "questions.json"), ("selection_manifest", "selection_manifest.json")]:
            shutil.copyfile(paths[key], out / name)
        shutil.copyfile(root / "protocol" / "freeze_manifest.json", out / "freeze_manifest.json")
        (out / "source_snapshot").mkdir()
        for name in source_names:
            shutil.copyfile(root / name, out / "source_snapshot" / name)
        import torch
        from nltk.tokenize.punkt import PunktTokenizer
        from sentence_transformers import SentenceTransformer
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        manifest["nltk_english_resource_sha256"] = nltk_resource_hashes()
        sentence_tokenizer = PunktTokenizer("english")
        model_start = time.perf_counter()
        model = SentenceTransformer(cfg["embedding"]["name"], revision=cfg["embedding"]["revision"],
                                    device=cfg["runtime"]["device"]).float().eval()
        manifest["timing"]["model_initialization_seconds"] = time.perf_counter() - model_start
        require(model.max_seq_length == cfg["embedding"]["max_input_tokens"], "Embedding token limit mismatch")
        manifest["model"] = dict(cfg["embedding"], device=cfg["runtime"]["device"], dtype="float32",
                                 embedding_dimension=model.get_sentence_embedding_dimension())
        manifest["torch_runtime"] = dict(num_threads=torch.get_num_threads(),
                                         num_interop_threads=torch.get_num_interop_threads())
        query_start = time.perf_counter()
        query_vectors = encode_checked([q["question"] for q in questions], model, cfg)
        manifest["timing"]["query_encoding_seconds"] = time.perf_counter() - query_start
        np.save(out / "query_embeddings.npy", query_vectors, allow_pickle=False)
        write_json(out / "run_manifest.json", manifest)
        summaries = {}
        for strategy in cfg["chunking"]["strategies"]:
            folder = out / strategy
            folder.mkdir()
            strategy_start = time.perf_counter()
            chunk_start = time.perf_counter()
            chunks = build_final_chunks(strategy, corpus, article_order, model, sentence_tokenizer, cfg)
            chunk_seconds = time.perf_counter() - chunk_start
            encode_start = time.perf_counter()
            vectors = encode_checked([c["text"] for c in chunks], model, cfg)
            encode_seconds = time.perf_counter() - encode_start
            write_json(folder / "chunks.json", chunks)
            np.save(folder / "chunk_embeddings.npy", vectors, allow_pickle=False)
            records, search_seconds = [], 0.0
            for index, (question, query) in enumerate(zip(questions, query_vectors), start=1):
                search_start = time.perf_counter()
                scores = np.sum(vectors * query, axis=1)
                require(np.isfinite(scores).all() and np.max(np.abs(scores)) <= 1.0001, "Invalid cosine scores")
                ranking = np.argsort(-scores, kind="stable")[:cfg["retrieval"]["top_k"]]
                search_seconds += time.perf_counter() - search_start
                retrieved = [{"rank": rank, "chunk": chunks[int(position)], "score": float(scores[position])}
                             for rank, position in enumerate(ranking, start=1)]
                hits = {f"hit_at_{k}": answer_span_hit_at_k(retrieved, question["article"],
                                                          question["answer_spans"], k=k)
                        for k in cfg["retrieval"]["report_hit_at"]}
                records.append(dict(question, retrieved_results=retrieved, retrieval_metrics=hits))
                if index % 100 == 0:
                    print(f"{strategy}: searched {index}/{len(questions)}", flush=True)
            require(len(records) == cfg["data"]["retrieval_questions"], "Incomplete retrieval records")
            require(sum(r["for_generation"] for r in records) == cfg["data"]["generation_questions"],
                    "Incomplete generation subset")
            summary = summarize(records, cfg)
            word_counts = [len(c["text"].split()) for c in chunks]
            summary["chunk_statistics"] = dict(n_chunks=len(chunks), max_words=max(word_counts),
                                                mean_words=sum(word_counts) / len(chunks))
            write_json(folder / "per_question.json", records)
            write_json(folder / "summary.json", summary)
            summaries[strategy] = summary
            manifest["timing"][strategy] = dict(chunking_including_unit_encoding_seconds=chunk_seconds,
                                               final_chunk_encoding_seconds=encode_seconds,
                                               similarity_and_ranking_seconds=search_seconds,
                                               total_strategy_seconds=time.perf_counter() - strategy_start)
            manifest["completed_strategies"].append(strategy)
            manifest["completed_records"] += len(records)
            write_json(out / "run_manifest.json", manifest)
            assert_unchanged(root, paths, manifest, source_names, cfg, load_freeze)
        require(manifest["completed_records"] == len(STRATEGIES) * cfg["data"]["retrieval_questions"],
                "Expected complete paired records for all strategies")
        write_json(out / "summary.json", summaries)
        assert_unchanged(root, paths, manifest, source_names, cfg, load_freeze)
        manifest.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_perf_seconds=time.perf_counter() - started,
                        output_sha256={str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*"))
                                       if p.is_file() and p.name != "run_manifest.json"})
        write_json(out / "run_manifest.json", manifest)
        print(json.dumps(summaries, indent=2))
        print("Saved FINAL retrieval:", out)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}",
                        stopped_at_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_perf_seconds=time.perf_counter() - started)
        write_json(out / "run_manifest.json", manifest)
        raise


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Frozen final retrieval; no language-model answer generation.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true", help="Validate structure/hashes only, without model/tokenizer instances")
    mode.add_argument("--execute", action="store_true", help="Execute only after real freeze/upload evidence is verified")
    args = parser.parse_args()
    cfg = read_json(root / "experiment_config.json")
    paths = input_paths(root, cfg)
    corpus, questions, article_order, gold_count = validate_inputs(cfg, paths)
    validate_runtime(cfg)
    selection = read_json(paths["selection_manifest"])
    if args.check_only:
        print(json.dumps(dict(purpose="final_retrieval_input_check", execution_requested=False,
                              config_status=cfg["status"], articles=len(article_order), questions=len(questions),
                              generation_questions=sum(q["for_generation"] for q in questions),
                              gold_annotations=gold_count, input_sha256={n: sha256(p) for n, p in paths.items()},
                              freeze_gate="Not evaluated; --execute requires real freeze evidence"), indent=2))
        return
    require(cfg["status"] == "frozen", "Execution locked: experiment_config.json is not frozen")
    from freeze_gate import load_freeze
    source_names = [Path(__file__).name, "freeze_gate.py", "check_final_inputs.py",
                    "run_development_retrieval.py", "chunkers.py", "semantic_chunker.py",
                    "sentence_splitter.py", "retrieval_metrics.py"]
    proof = load_freeze(root, cfg, paths, source_names)
    execute(root, cfg, paths, corpus, questions, article_order, selection, source_names, proof, load_freeze)


if __name__ == "__main__":
    main()
