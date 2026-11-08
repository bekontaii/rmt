import hashlib
import json
import platform
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from itertools import islice
from pathlib import Path

from chunkers import fixed_chunks
from semantic_chunker import prepare_units


A3_ARTICLES = {"Warsaw", "Teacher", "Martin_Luther", "Southern_California",
               "Sky_(United_Kingdom)", "Apollo_program", "European_Union_law",
               "Ctenophora", "Doctor_Who", "Prime_number"}
EMBED_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBED_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
GEN_NAME = "google/flan-t5-large"
GEN_REVISION = "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"


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
    require(root.resolve() in path.parents, "Input path escapes this project")
    return path


def validate_inputs(cfg, paths):
    manifest = read_json(paths["selection_manifest"])
    corpus, questions = read_json(paths["corpus"]), read_json(paths["questions"])
    raw = read_json(paths["raw_dataset"])
    require(raw["version"] == "1.1", "Expected SQuAD version 1.1")
    data, chunk, emb, gen = cfg["data"], cfg["chunking"], cfg["embedding"], cfg["generation"]
    require(cfg["seed"] == manifest["seed"] == 42, "Selection seed mismatch")
    require(sha256(paths["raw_dataset"]) == data["dataset_sha256"] == manifest["dataset_sha256"],
            "Raw dataset SHA256 mismatch")
    require(data["articles"] == 20 and data["retrieval_questions"] == 2000
            and data["generation_questions"] == 1000
            and data["retrieval_questions_per_article"] == manifest["retrieval_questions_per_article"] == 100
            and data["generation_questions_per_article"] == manifest["generation_questions_per_article"] == 50,
            "Expected the planned 20/2000/1000 sample with 100/50 questions per article")
    titles, ids, gen_ids = manifest["article_titles"], manifest["retrieval_question_ids"], manifest["generation_question_ids"]
    require(len(titles) == len(set(titles)) == 20 and list(corpus) == titles, "Article identities/order mismatch")
    require(len(manifest["excluded_a3_articles"]) == 10
            and set(manifest["excluded_a3_articles"]) == A3_ARTICLES
            and not set(titles) & A3_ARTICLES, "Final sample overlaps or changes the A3 exclusion list")
    require(len(ids) == len(set(ids)) == 2000 and len(gen_ids) == len(set(gen_ids)) == 1000
            and set(gen_ids) <= set(ids), "Invalid question IDs/subset")
    gen_set = set(gen_ids)
    selected_set = set(ids)
    require([i for i in ids if i in gen_set] == gen_ids, "Generation ID order mismatch")
    require(len(questions) == 2000 and [q["id"] for q in questions] == ids, "Question records/order mismatch")
    raw_articles = {a["title"]: a for a in raw["data"]}
    available = {}
    for title in titles:
        article = raw_articles[title]
        require(corpus[title] == "\n\n".join(p["context"] for p in article["paragraphs"]),
                f"Corpus differs from complete raw article: {title}")
        offset = 0
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if qa["id"] not in selected_set:
                    continue
                answers, spans = [], []
                for answer in qa["answers"]:
                    start, text = answer["answer_start"], answer["text"]
                    require(type(start) is int and 0 <= start < start + len(text) <= len(paragraph["context"])
                            and paragraph["context"][start:start + len(text)] == text, "Invalid raw annotation")
                    answers.append(text)
                    spans.append([offset + start, offset + start + len(text)])
                require(qa["id"] not in available, "Duplicate raw question ID")
                available[qa["id"]] = dict(id=qa["id"], article=title, question=qa["question"],
                                           answers=answers, answer_spans=spans, for_generation=qa["id"] in gen_set)
            offset += len(paragraph["context"]) + 2
    for q in questions:
        require(type(q["for_generation"]) is bool and q == available[q["id"]],
                f"Selected question/text/answers/global spans mismatch: {q['id']}")
        require(bool(q["question"].strip()) and bool(q["answers"]), "Empty question or gold references")
        for answer, (start, end) in zip(q["answers"], q["answer_spans"]):
            require(corpus[q["article"]][start:end] == answer, "Global gold/source mismatch")
    require(Counter(q["article"] for q in questions) == Counter({t: 100 for t in titles})
            and Counter(q["article"] for q in questions if q["for_generation"]) == Counter({t: 50 for t in titles}),
            "Per-article question counts mismatch")
    require(chunk["strategies"] == ["fixed", "overlap", "semantic"] and chunk["max_words"] == 100
            and chunk["overlap_words"] == 20 and chunk["semantic_percentile"] == 90
            and chunk["word_count_rule"] == "whitespace" and chunk["semantic_threshold_scope"] == "article"
            and chunk["sentence_splitter"] == "nltk_punkt_english", "Unsupported chunking profile")
    require(emb["name"] == EMBED_NAME and emb["revision"] == EMBED_REVISION
            and emb["max_input_tokens"] == 384 and emb["overflow_policy"] == "error"
            and emb["normalize_embeddings"] is True, "Unsupported embedding profile")
    require(gen["name"] == GEN_NAME and gen["revision"] == GEN_REVISION
            and gen["max_input_tokens"] == 512 and gen["context_top_k"] == 3
            and gen["context_policy"] == "ranked_prefix_whole_words"
            and "{question}" in gen["prompt_template"] and "{context}" in gen["prompt_template"],
            "Unsupported generation prompt profile")
    require(cfg["runtime"] == {"device": "cpu", "dtype": "float32"}
            and emb["batch_size"] == 32, "Expected CPU float32 and embedding batch size 32")
    require(gen["max_new_tokens"] == 32 and gen["do_sample"] is False
            and gen["num_beams"] == 1 and gen["batch_size"] == 1,
            "Expected greedy single-answer generation with a 32-token output limit")
    ret = cfg["retrieval"]
    require(ret["search_scope"] == "all_selected_articles" and ret["top_k"] == 5
            and ret["report_hit_at"] == [1, 3, 5]
            and ret["tie_break"] == "original_chunk_order", "Unexpected retrieval settings")
    return corpus, questions, titles, sum(len(q["answers"]) for q in questions)


def check_piece(piece, article, text, max_words):
    start, end = piece["start"], piece["end"]
    require(piece["article"] == article and type(start) is int and type(end) is int
            and 0 <= start < end <= len(text) and piece["text"] == text[start:end]
            and 1 <= len(piece["text"].split()) <= max_words, "Unit/chunk/source mismatch")


def candidate_spans(article, text, units, max_words):
    for i in range(len(units)):
        start = units[i]["start"]
        for j in range(i, len(units)):
            end = units[j]["end"]
            candidate = text[start:end]
            if len(candidate.split()) > max_words:
                break
            yield candidate, dict(article=article, start=start, end=end)


def new_stats():
    return dict(count=0, max_tokens=0, over_limit=0, overflow_examples=[])


def audit_tokens(items, tokenizer, limit, stats):
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, 64))
        if not batch:
            break
        encoded = tokenizer([text for text, _ in batch], add_special_tokens=True,
                            truncation=False, max_length=limit, padding=False)
        require(len(encoded["input_ids"]) == len(batch), "Tokenizer batch size mismatch")
        for ids, (_, location) in zip(encoded["input_ids"], batch):
            count = len(ids)
            stats["count"] += 1
            stats["max_tokens"] = max(stats["max_tokens"], count)
            if count > limit:
                stats["over_limit"] += 1
                if len(stats["overflow_examples"]) < 5:
                    stats["overflow_examples"].append(dict(location, tokens=count, limit=limit))


def versions():
    result = {}
    for name in ["transformers", "tokenizers", "sentencepiece", "nltk", "sentence-transformers", "numpy"]:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def main():
    root = Path(__file__).resolve().parent
    config_path = root / "experiment_config.json"
    cfg = read_json(config_path)
    paths = dict(config=config_path, raw_dataset=inside(root, cfg["data"]["dataset_path"]),
                 selection_manifest=inside(root, cfg["data"]["selection_manifest_path"]),
                 corpus=inside(root, cfg["data"]["corpus_path"]), questions=inside(root, cfg["data"]["questions_path"]))
    out = root / "development" / "preflight_runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True, exist_ok=False)
    names = [Path(__file__).name, "chunkers.py", "semantic_chunker.py", "sentence_splitter.py"]
    report = dict(purpose="final_input_preflight", status="running", prediction_runs=0, model_embedding_runs=0,
                  started_at_utc=datetime.now(timezone.utc).isoformat(), python_version=platform.python_version(),
                  package_versions=versions(), input_sha256={n: sha256(p) for n, p in paths.items()},
                  source_sha256={n: sha256(root / n) for n in names}, token_stats={})
    write_json(out / "report.json", report)
    try:
        shutil.copyfile(config_path, out / "config_snapshot.json")
        shutil.copyfile(paths["selection_manifest"], out / "selection_manifest.json")
        (out / "source_snapshot").mkdir()
        for name in names:
            shutil.copyfile(root / name, out / "source_snapshot" / name)
        corpus, questions, titles, gold_count = validate_inputs(cfg, paths)
        report.update(articles_checked=len(titles), retrieval_questions_checked=len(questions),
                      generation_questions_checked=sum(q["for_generation"] for q in questions),
                      gold_annotations_checked=gold_count)
        import nltk
        from nltk.tokenize.punkt import PunktTokenizer
        from transformers import AutoTokenizer
        resources = nltk.data.find("tokenizers/punkt_tab/english/")
        resource_names = ["abbrev_types.txt", "collocations.tab", "ortho_context.tab", "sent_starters.txt"]
        report["nltk_english_resource_sha256"] = {}
        for name in resource_names:
            with resources.join(name).open() as stream:
                report["nltk_english_resource_sha256"][name] = hashlib.sha256(stream.read()).hexdigest()
        sentence_tokenizer = PunktTokenizer("english")
        emb, gen, chunk = cfg["embedding"], cfg["generation"], cfg["chunking"]
        print("Loading tokenizers only; no model weights or predictions.", flush=True)
        embedding_tokenizer = AutoTokenizer.from_pretrained(emb["name"], revision=emb["revision"], use_fast=True)
        generation_tokenizer = AutoTokenizer.from_pretrained(gen["name"], revision=gen["revision"],
                                                             use_fast=False, legacy=True)
        require(embedding_tokenizer.is_fast and not generation_tokenizer.is_fast, "Tokenizer backend mismatch")
        report["tokenizers"] = dict(embedding=dict(name=emb["name"], revision=emb["revision"], use_fast=True),
                                     generation=dict(name=gen["name"], revision=gen["revision"], use_fast=False, legacy=True))
        stats = report["token_stats"]
        for name in ["embedding_questions", "fixed_chunks", "overlap_chunks", "semantic_candidates", "minimum_generation_prompts"]:
            stats[name] = new_stats()
        audit_tokens(((q["question"], dict(id=q["id"], article=q["article"])) for q in questions),
                     embedding_tokenizer, emb["max_input_tokens"], stats["embedding_questions"])
        minimal = ((gen["prompt_template"].format(question=q["question"], context=""),
                    dict(id=q["id"], article=q["article"])) for q in questions if q["for_generation"])
        audit_tokens(minimal, generation_tokenizer, gen["max_input_tokens"], stats["minimum_generation_prompts"])
        report["prepared_units_checked"] = 0
        for number, article in enumerate(titles, start=1):
            text = corpus[article]
            print(f"{number}/20: {article}", flush=True)
            for label, overlap in [("fixed_chunks", 0), ("overlap_chunks", chunk["overlap_words"])]:
                pieces = fixed_chunks(article, text, size=chunk["max_words"], overlap=overlap)
                for piece in pieces:
                    check_piece(piece, article, text, chunk["max_words"])
                audit_tokens(((p["text"], dict(article=article, start=p["start"], end=p["end"])) for p in pieces),
                             embedding_tokenizer, emb["max_input_tokens"], stats[label])
            units = prepare_units(article, text, sentence_tokenizer, max_words=chunk["max_words"])
            previous_end = 0
            for unit in units:
                check_piece(unit, article, text, chunk["max_words"])
                require(unit["start"] >= previous_end and not re.search(r"\S", text[previous_end:unit["start"]]),
                        "Prepared units overlap or leave non-whitespace source characters uncovered")
                previous_end = unit["end"]
            require(not re.search(r"\S", text[previous_end:]), "Prepared units omit the source tail")
            report["prepared_units_checked"] += len(units)
            audit_tokens(candidate_spans(article, text, units, chunk["max_words"]),
                         embedding_tokenizer, emb["max_input_tokens"], stats["semantic_candidates"])
            write_json(out / "report.json", report)
        require(stats["embedding_questions"]["count"] == 2000
                and stats["minimum_generation_prompts"]["count"] == 1000, "Token audit question count mismatch")
        require(all(sha256(p) == report["input_sha256"][n] for n, p in paths.items()), "Inputs changed during preflight")
        require(all(sha256(root / n) == h for n, h in report["source_sha256"].items()), "Source code changed")
        print(json.dumps(stats, indent=2), flush=True)
        require(not any(s["over_limit"] for s in stats.values()),
                "Token overflows found; inspect report.json. No data or method was automatically changed.")
        report.update(status="passed", completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(out / "report.json", report)
        print("Preflight PASSED. Saved:", out)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}",
                      stopped_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(out / "report.json", report)
        print("Preflight FAILED. Saved:", out, flush=True)
        raise


if __name__ == "__main__":
    main()
