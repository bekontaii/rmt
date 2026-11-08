import json
from pathlib import Path


def validate_span(span):
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        raise ValueError("Span must contain [start, end]")

    start, end = span

    if type(start) is not int or type(end) is not int:
        raise ValueError("Span positions must be integers")

    if not 0 <= start < end:
        raise ValueError("Span must satisfy 0 <= start < end")

    return start, end


def answer_span_hit_at_k(
    retrieved_results,
    article,
    answer_spans,
    k=3,
):
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")

    if len(retrieved_results) < k:
        raise ValueError(
            f"Hit@{k} requires at least {k} saved results; "
            f"only {len(retrieved_results)} are available"
        )

    if not answer_spans:
        raise ValueError("At least one answer span is required")

    checked_spans = [
        validate_span(span)
        for span in answer_spans
    ]

    hit = False

    for result in retrieved_results[:k]:
        if not isinstance(result, dict):
            raise ValueError("Retrieved result must be a dictionary")

        chunk = result.get("chunk")

        if not isinstance(chunk, dict):
            raise ValueError("Retrieved result must contain a chunk")

        if not isinstance(chunk.get("article"), str):
            raise ValueError("Chunk must contain an article name")

        chunk_start, chunk_end = validate_span([
            chunk.get("start"),
            chunk.get("end"),
        ])

        if chunk["article"] != article:
            continue

        for answer_start, answer_end in checked_spans:
            if (
                chunk_start <= answer_start
                and answer_end <= chunk_end
            ):
                hit = True

    return float(hit)


def run_checks():
    # Искусственные примеры для проверки формулы.
    def item(article, start, end):
        return {
            "chunk": {
                "article": article,
                "start": start,
                "end": end,
            }
        }

    article = "Programming"
    spans = [[0, 6]]

    wrong_article = item("Other_article", 0, 30)
    left_half = item(article, 0, 3)
    right_half = item(article, 3, 30)
    whole_answer = item(article, 0, 30)

    # Та же позиция в другой статье не считается успехом.
    assert answer_span_hit_at_k(
        [wrong_article], article, spans, k=1
    ) == 0.0

    # Часть ответа не считается полным ответом.
    assert answer_span_hit_at_k(
        [left_half], article, spans, k=1
    ) == 0.0

    # Два частичных чанка не заменяют один полный.
    assert answer_span_hit_at_k(
        [left_half, right_half], article, spans, k=2
    ) == 0.0

    # Полный ответ находится только на втором месте.
    ranked_results = [wrong_article, whole_answer]

    assert answer_span_hit_at_k(
        ranked_results, article, spans, k=1
    ) == 0.0

    assert answer_span_hit_at_k(
        ranked_results, article, spans, k=2
    ) == 1.0

    # Достаточно покрытия любого из эталонных spans.
    assert answer_span_hit_at_k(
        [item(article, 10, 20)],
        article,
        [[0, 6], [10, 16]],
        k=1,
    ) == 1.0

    print("Retrieval metric checks: OK")


if __name__ == "__main__":
    run_checks()

    project_directory = Path(__file__).resolve().parent

    results_directory = (
        project_directory
        / "development"
        / "toy_runs"
    )

    saved_files = sorted(
        results_directory.glob("rag_demo_*.json")
    )

    if not saved_files:
        raise FileNotFoundError(
            f"No demo JSON files found in {results_directory}. "
            "Run python rag_demo.py first."
        )

    # Читаем последнюю запись учебного запуска.
    result_path = saved_files[-1]

    record = json.loads(
        result_path.read_text(encoding="utf-8")
    )

    if record.get("run_type") != "toy_development_demo":
        raise ValueError("Expected a toy development demo")

    # Эталонные позиции именно для учебного вопроса про Python.
    correct_article = "Programming"
    answer_spans = [[0, 6]]

    source_text = record["documents"][correct_article]

    if (
        source_text[0:6] != "Python"
        or "Python" not in record["reference_answers"]
    ):
        raise ValueError(
            "This demo expects the Python question "
            "and its original Programming document"
        )

    retrieved_results = record["retrieved_results"]

    hit_at_1 = answer_span_hit_at_k(
        retrieved_results,
        correct_article,
        answer_spans,
        k=1,
    )

    hit_at_3 = answer_span_hit_at_k(
        retrieved_results,
        correct_article,
        answer_spans,
        k=3,
    )

    print("\nFile:", result_path)
    print("Question:", record["question"])
    print("Generated answer:", record["prediction"])
    print("Reference answer: Python")
    print("Answer span: [0:6]")

    print(f"\nHit@1: {hit_at_1:.3f}")
    print(f"Hit@3: {hit_at_3:.3f}")

    print(
        f"Stored Exact Match: "
        f"{record['metrics']['exact_match']:.3f}"
    )
    print(
        f"Stored Token F1: "
        f"{record['metrics']['f1']:.3f}"
    )