import hashlib
import json
import random
from pathlib import Path


DEV_ARTICLES = [
    "Warsaw",
    "Martin_Luther",
    "Teacher",
]

QUESTIONS_PER_ARTICLE = 3


def prepare_article(article):
    paragraphs = article["paragraphs"]

    full_text = "\n\n".join(
        paragraph["context"]
        for paragraph in paragraphs
    )

    questions = []
    offset = 0

    for paragraph in paragraphs:
        context = paragraph["context"]

        for qa in paragraph["qas"]:
            answers = []
            answer_spans = []

            for answer in qa["answers"]:
                start = offset + answer["answer_start"]
                end = start + len(answer["text"])

                answers.append(answer["text"])
                answer_spans.append([start, end])

            questions.append({
                "id": qa["id"],
                "article": article["title"],
                "question": qa["question"],
                "answers": answers,
                "answer_spans": answer_spans,
                "for_generation": True,
            })

        offset += len(context) + 2

    return full_text, questions


def main():
    project_directory = Path(__file__).resolve().parent

    config = json.loads(
        (project_directory / "experiment_config.json").read_text(
            encoding="utf-8"
        )
    )

    data_config = config["data"]

    manifest_path = (
        project_directory
        / data_config["selection_manifest_path"]
    )

    manifest_bytes = manifest_path.read_bytes()
    final_manifest = json.loads(manifest_bytes)

    dataset_path = (
        project_directory
        / data_config["dataset_path"]
    )

    dataset_bytes = dataset_path.read_bytes()

    dataset_sha256 = hashlib.sha256(
        dataset_bytes
    ).hexdigest()

    if not (
        dataset_sha256
        == data_config["dataset_sha256"]
        == final_manifest["dataset_sha256"]
    ):
        raise ValueError("Dataset SHA256 mismatch")

    dataset = json.loads(dataset_bytes)

    if dataset["version"] != "1.1":
        raise ValueError("Expected SQuAD version 1.1")

    old_a3_articles = set(
        final_manifest["excluded_a3_articles"]
    )

    if not set(DEV_ARTICLES).issubset(old_a3_articles):
        raise ValueError(
            "Development articles must come from Assignment 3"
        )

    article_overlap = (
        set(DEV_ARTICLES)
        & set(final_manifest["article_titles"])
    )

    if article_overlap:
        raise ValueError(
            f"Development/final article overlap: {article_overlap}"
        )

    articles_by_title = {
        article["title"]: article
        for article in dataset["data"]
    }

    rng = random.Random(config["seed"])

    corpus = {}
    questions = []

    for title in DEV_ARTICLES:
        if title not in articles_by_title:
            raise ValueError(f"Article not found: {title}")

        full_text, available_questions = prepare_article(
            articles_by_title[title]
        )

        corpus[title] = full_text

        sampled_questions = rng.sample(
            available_questions,
            QUESTIONS_PER_ARTICLE,
        )

        questions.extend(sampled_questions)

    question_ids = [
        question["id"]
        for question in questions
    ]

    expected_count = (
        len(DEV_ARTICLES) * QUESTIONS_PER_ARTICLE
    )

    if (
        len(question_ids) != expected_count
        or len(set(question_ids)) != expected_count
    ):
        raise ValueError("Unexpected or duplicate question IDs")

    question_overlap = (
        set(question_ids)
        & set(final_manifest["retrieval_question_ids"])
    )

    if question_overlap:
        raise ValueError(
            "Development questions overlap with the final sample"
        )

    checked_spans = 0

    for question in questions:
        source = corpus[question["article"]]

        if not question["answers"]:
            raise ValueError("Question has no reference answers")

        for answer, span in zip(
            question["answers"],
            question["answer_spans"],
        ):
            start, end = span

            if not (
                answer
                and 0 <= start < end <= len(source)
                and source[start:end] == answer
            ):
                raise ValueError(
                    f"Invalid answer span: {question['id']}"
                )

            checked_spans += 1

    development_manifest = {
        "purpose": "development_pilot",
        "dataset_sha256": dataset_sha256,
        "final_selection_manifest_sha256": hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
        "seed": config["seed"],
        "article_titles": DEV_ARTICLES,
        "questions_per_article": QUESTIONS_PER_ARTICLE,
        "sampling_rule": (
            "One local Random(seed); sample questions "
            "sequentially in the recorded article order"
        ),
        "retrieval_question_ids": question_ids,
        "generation_question_ids": question_ids,
        "final_article_overlap": sorted(article_overlap),
        "final_question_overlap": sorted(question_overlap),
    }

    output_directory = (
        project_directory
        / "development"
        / "pilot_data"
    )

    outputs = {
        "corpus.json": corpus,
        "questions.json": questions,
        "selection_manifest.json": development_manifest,
    }

    serialized = {
        name: json.dumps(
            content,
            ensure_ascii=False,
            indent=2,
        ) + "\n"
        for name, content in outputs.items()
    }

    # Существующие отличающиеся данные не перезаписываем.
    for name, content in serialized.items():
        path = output_directory / name

        if path.exists():
            existing = path.read_text(encoding="utf-8")

            if existing != content:
                raise FileExistsError(
                    f"Different development data already exist: {path}"
                )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, content in serialized.items():
        path = output_directory / name

        if not path.exists():
            path.write_text(
                content,
                encoding="utf-8",
            )

    print("Development articles:", len(corpus))
    print("Retrieval questions:", len(questions))
    print(
        "Generation questions:",
        sum(q["for_generation"] for q in questions),
    )
    print("Final article overlap:", len(article_overlap))
    print("Final question overlap:", len(question_overlap))
    print("Answer spans checked:", checked_spans)

    print("\nExample question:", questions[0]["question"])
    print("Reference answers:", questions[0]["answers"])

    print("\nSaved to:", output_directory)
    print("Development preparation passed.")


if __name__ == "__main__":
    main()