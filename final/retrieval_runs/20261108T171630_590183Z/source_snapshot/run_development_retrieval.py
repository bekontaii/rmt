import hashlib
import json
import platform
import shutil
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from chunkers import fixed_chunks
from retrieval_metrics import answer_span_hit_at_k
from semantic_chunker import MODEL_NAME, MODEL_REVISION, semantic_chunks


ARTICLES = ["Warsaw", "Martin_Luther", "Teacher"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_inputs(root):
    paths = {"config": root / "experiment_config.json"}
    cfg = read_json(paths["config"])
    base = root / "development" / "pilot_data"
    paths.update(corpus=base / "corpus.json", questions=base / "questions.json",
                 development_manifest=base / "selection_manifest.json",
                 final_manifest=root / cfg["data"]["selection_manifest_path"],
                 raw_dataset=root / cfg["data"]["dataset_path"])
    corpus, questions = read_json(paths["corpus"]), read_json(paths["questions"])
    dev, final = read_json(paths["development_manifest"]), read_json(paths["final_manifest"])
    require(dev["purpose"] == "development_pilot", "Only development pilot data are allowed")
    require(dev["seed"] == cfg["seed"] == final["seed"] == 42, "Seed mismatch")
    require(dev["article_titles"] == ARTICLES and list(corpus) == ARTICLES,
            "Expected the three development articles in their recorded order")
    require(set(ARTICLES) <= set(final["excluded_a3_articles"]), "Not excluded A3 articles")
    require(not set(ARTICLES) & set(final["article_titles"]), "Final article overlap")
    require(sha256(paths["raw_dataset"]) == cfg["data"]["dataset_sha256"]
            == dev["dataset_sha256"] == final["dataset_sha256"], "Raw dataset hash mismatch")
    require(sha256(paths["final_manifest"]) == dev["final_selection_manifest_sha256"],
            "The final selection manifest has changed since development preparation")
    ids = [q["id"] for q in questions]
    require(len(ids) == len(set(ids)) == 9, "Expected nine unique development questions")
    require(ids == dev["retrieval_question_ids"] == dev["generation_question_ids"],
            "Development question identities/order do not match the manifest")
    require(not set(ids) & set(final["retrieval_question_ids"]), "Final question overlap")
    require(dev["questions_per_article"] == 3
            and Counter(q["article"] for q in questions) == Counter({a: 3 for a in ARTICLES}),
            "Expected three questions per development article")
    for q in questions:
        require(q["for_generation"] is True and isinstance(q["question"], str)
                and bool(q["question"].strip()), "Invalid development question")
        require(bool(q["answers"]) and len(q["answers"]) == len(q["answer_spans"]),
                "Missing or unpaired gold answers/spans")
        for answer, (start, end) in zip(q["answers"], q["answer_spans"]):
            require(type(start) is int and type(end) is int
                    and 0 <= start < end <= len(corpus[q["article"]])
                    and corpus[q["article"]][start:end] == answer, "Gold span mismatch")
    return cfg, corpus, questions, paths


def validate_config(cfg):
    chunk, emb, ret = cfg["chunking"], cfg["embedding"], cfg["retrieval"]
    require(cfg["runtime"] == {"device": "cpu", "dtype": "float32"},
            "This development runner supports CPU float32 only")
    require(emb["name"] == MODEL_NAME and emb["revision"] == MODEL_REVISION,
            "Embedding name/revision differs from semantic_chunker.py")
    require(emb["normalize_embeddings"] is True and emb["overflow_policy"] == "error"
            and emb["max_input_tokens"] == 384, "Unsupported embedding policy")
    require(type(emb["batch_size"]) is int and emb["batch_size"] == 32,
            "Current semantic helper uses batch32; this runner requires the same configured batch size")
    require(chunk["strategies"] == ["fixed", "overlap", "semantic"]
            and chunk["word_count_rule"] == "whitespace"
            and chunk["semantic_threshold_scope"] == "article"
            and chunk["sentence_splitter"] == "nltk_punkt_english", "Unsupported chunking policy")
    require(type(chunk["max_words"]) is int and chunk["max_words"] > 0
            and type(chunk["overlap_words"]) is int
            and 0 <= chunk["overlap_words"] < chunk["max_words"]
            and 0 <= chunk["semantic_percentile"] <= 100, "Invalid chunking parameters")
    require(ret["search_scope"] == "all_selected_articles"
            and ret["tie_break"] == "original_chunk_order"
            and ret["top_k"] == 5 and ret["report_hit_at"] == [1, 3, 5],
            "Expected global top5 retrieval with stable ties and Hit@1/3/5")


def encode_checked(texts, model, cfg):
    emb = cfg["embedding"]
    require(bool(texts), "Cannot encode an empty collection")
    tokenized = model.tokenizer(texts, add_special_tokens=True, truncation=False,
                                max_length=emb["max_input_tokens"])
    longest = max(len(ids) for ids in tokenized["input_ids"])
    require(longest <= emb["max_input_tokens"],
            f"Embedding input has {longest} tokens; limit is {emb['max_input_tokens']}")
    vectors = model.encode(texts, batch_size=emb["batch_size"],
                           normalize_embeddings=emb["normalize_embeddings"],
                           convert_to_numpy=True, show_progress_bar=True)
    require(vectors.shape == (len(texts), model.get_sentence_embedding_dimension()),
            "Unexpected embedding matrix shape")
    require(vectors.dtype == np.float32 and np.isfinite(vectors).all(),
            "Embeddings must be finite float32")
    norms = np.sqrt(np.sum(vectors * vectors, axis=1))
    require(np.allclose(norms, 1.0, atol=1e-5, rtol=1e-5), "Embeddings are not normalized")
    return vectors


def build_chunks(strategy, corpus, model, sentence_tokenizer, cfg):
    settings, chunks = cfg["chunking"], []
    for article in ARTICLES:
        print(f"  {strategy}: {article}", flush=True)
        text = corpus[article]
        if strategy == "semantic":
            pieces = semantic_chunks(article, text, model, sentence_tokenizer,
                                     max_words=settings["max_words"],
                                     breakpoint_percentile=settings["semantic_percentile"])
        else:
            overlap = settings["overlap_words"] if strategy == "overlap" else 0
            pieces = fixed_chunks(article, text, size=settings["max_words"], overlap=overlap)
        for piece in pieces:
            start, end = piece["start"], piece["end"]
            require(piece["article"] == article and type(start) is int and type(end) is int
                    and 0 <= start < end <= len(text) and piece["text"] == text[start:end],
                    "Chunk/source offsets do not match")
            require(1 <= len(piece["text"].split()) <= settings["max_words"], "Invalid word cap")
            chunks.append(dict(piece, chunk_id=f"{strategy}:{len(chunks):05d}"))
    require(len(chunks) >= cfg["retrieval"]["top_k"], "Too few chunks for top5")
    return chunks


def package_versions():
    result = {}
    for name in ["numpy", "torch", "transformers", "sentence-transformers", "nltk",
                 "tokenizers", "huggingface-hub"]:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def main():
    root = Path(__file__).resolve().parent
    cfg, corpus, questions, paths = load_inputs(root)
    validate_config(cfg)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = root / "development" / "retrieval_runs" / run_id
    out.mkdir(parents=True, exist_ok=False)
    source_names = [Path(__file__).name, "chunkers.py", "semantic_chunker.py",
                    "sentence_splitter.py", "retrieval_metrics.py"]
    manifest = {"purpose": "development_retrieval", "status": "running", "run_id": run_id,
                "started_at_utc": datetime.now(timezone.utc).isoformat(), "seed": cfg["seed"],
                "article_titles": ARTICLES, "question_ids": [q["id"] for q in questions],
                "input_sha256": {name: sha256(path) for name, path in paths.items()},
                "source_sha256": {name: sha256(root / name) for name in source_names},
                "package_versions": package_versions(), "python_version": platform.python_version(),
                "platform": platform.platform(), "completed_strategies": []}
    write_json(out / "run_manifest.json", manifest)
    try:
        for key, filename in [("config", "config_snapshot.json"), ("questions", "questions.json"),
                              ("corpus", "corpus.json"), ("development_manifest", "selection_manifest.json")]:
            shutil.copyfile(paths[key], out / filename)
        source_folder = out / "source_snapshot"
        source_folder.mkdir()
        for name in source_names:
            shutil.copyfile(root / name, source_folder / name)
        import torch
        from nltk.tokenize.punkt import PunktTokenizer
        from sentence_transformers import SentenceTransformer
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        print("Loading MPNet...", flush=True)
        model = SentenceTransformer(cfg["embedding"]["name"], revision=cfg["embedding"]["revision"],
                                    device=cfg["runtime"]["device"]).float().eval()
        require(model.max_seq_length == cfg["embedding"]["max_input_tokens"], "Model token limit mismatch")
        sentence_tokenizer = PunktTokenizer("english")
        query_vectors = encode_checked([q["question"] for q in questions], model, cfg)
        np.save(out / "query_embeddings.npy", query_vectors, allow_pickle=False)
        summaries = {}
        for strategy in cfg["chunking"]["strategies"]:
            print("Development retrieval:", strategy, flush=True)
            folder = out / strategy
            folder.mkdir()
            chunks = build_chunks(strategy, corpus, model, sentence_tokenizer, cfg)
            vectors = encode_checked([c["text"] for c in chunks], model, cfg)
            write_json(folder / "chunks.json", chunks)
            np.save(folder / "chunk_embeddings.npy", vectors, allow_pickle=False)
            records = []
            for question, query in zip(questions, query_vectors):
                scores = np.sum(vectors * query, axis=1)
                require(np.isfinite(scores).all() and np.max(np.abs(scores)) <= 1.0001,
                        "Invalid cosine scores")
                ranking = np.argsort(-scores, kind="stable")[:cfg["retrieval"]["top_k"]]
                retrieved = [{"rank": rank, "chunk": chunks[int(index)], "score": float(scores[index])}
                             for rank, index in enumerate(ranking, start=1)]
                hits = {f"hit_at_{k}": answer_span_hit_at_k(retrieved, question["article"],
                                                          question["answer_spans"], k=k)
                        for k in cfg["retrieval"]["report_hit_at"]}
                records.append(dict(question, retrieved_results=retrieved, retrieval_metrics=hits))
            summary = {"n_questions": len(records), "n_chunks": len(chunks)}
            for k in cfg["retrieval"]["report_hit_at"]:
                name = f"hit_at_{k}"
                summary[name] = sum(r["retrieval_metrics"][name] for r in records) / len(records)
            write_json(folder / "per_question.json", records)
            write_json(folder / "summary.json", summary)
            summaries[strategy] = summary
            manifest["completed_strategies"].append(strategy)
            write_json(out / "run_manifest.json", manifest)
        write_json(out / "summary.json", summaries)
        require(all(sha256(path) == manifest["input_sha256"][name] for name, path in paths.items()),
                "An input changed during the run")
        require(all(sha256(root / name) == value for name, value in manifest["source_sha256"].items()),
                "A source file changed during the run")
        manifest.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        output_sha256={str(p.relative_to(out)): sha256(p)
                                       for p in sorted(out.rglob("*"))
                                       if p.is_file() and p.name != "run_manifest.json"})
        write_json(out / "run_manifest.json", manifest)
        print(json.dumps(summaries, indent=2))
        print("Saved development retrieval:", out)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}",
                        stopped_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(out / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
