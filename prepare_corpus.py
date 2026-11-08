import hashlib
import json
from pathlib import Path

selection = json.loads(
    Path("data/selection_manifest.json").read_text(encoding="utf-8")
)

raw_data = Path("data/dev-v1.1.json").read_bytes()

assert hashlib.sha256(raw_data).hexdigest() == selection["dataset_sha256"]

dataset = json.loads(raw_data)

articles_by_title = {
    article["title"]: article
    for article in dataset["data"]
}

retrieval_order = selection["retrieval_question_ids"]
retrieval_ids = set(retrieval_order)
generation_ids = set(selection["generation_question_ids"])

assert len(retrieval_order) == len(retrieval_ids) == 2000
assert len(selection["generation_question_ids"]) == len(generation_ids) == 1000
assert generation_ids.issubset(retrieval_ids)

corpus = {}
questions_by_id = {}

for title in selection["article_titles"]:
    article = articles_by_title[title]
    parts = []
    offset = 0

    for paragraph in article["paragraphs"]:
        context = paragraph["context"]
        parts.append(context)

        for qa in paragraph["qas"]:
            if qa["id"] not in retrieval_ids:
                continue

            assert qa["id"] not in questions_by_id

            spans = []

            for answer in qa["answers"]:
                start = offset + answer["answer_start"]
                end = start + len(answer["text"])
                spans.append([start, end])

            questions_by_id[qa["id"]] = {
                "id": qa["id"],
                "article": title,
                "question": qa["question"],
                "answers": [
                    answer["text"] for answer in qa["answers"]
                ],
                "answer_spans": spans,
                "for_generation": qa["id"] in generation_ids,
            }

        offset += len(context) + 2

    corpus[title] = "\n\n".join(parts)

questions = [
    questions_by_id[qid] for qid in retrieval_order
]

assert len(corpus) == 20
assert len(questions) == 2000
assert sum(q["for_generation"] for q in questions) == 1000

checked_spans = 0

for question in questions:
    article_text = corpus[question["article"]]

    for answer, span in zip(
        question["answers"],
        question["answer_spans"],
    ):
        start, end = span
        assert article_text[start:end] == answer
        checked_spans += 1

Path("data/corpus.json").write_text(
    json.dumps(corpus, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

Path("data/questions.json").write_text(
    json.dumps(questions, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("Articles prepared:", len(corpus))
print("Questions prepared:", len(questions))
print("Generation questions:", sum(q["for_generation"] for q in questions))
print("Answer spans checked:", checked_spans)
print("Corpus words:", sum(len(text.split()) for text in corpus.values()))
print("Corpus preparation passed.")