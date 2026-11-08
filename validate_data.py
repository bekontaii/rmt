import hashlib
import json
from pathlib import Path

dataset_path = Path("data/dev-v1.1.json")
selection_path = Path("data/selection_manifest.json")

raw_data = dataset_path.read_bytes()
dataset = json.loads(raw_data)

selection = json.loads(
    selection_path.read_text(encoding="utf-8")
)

dataset_hash = hashlib.sha256(raw_data).hexdigest()

assert dataset_hash == selection["dataset_sha256"]

selected_ids = set(selection["retrieval_question_ids"])

assert len(selection["retrieval_question_ids"]) == len(selected_ids) == 2000

found_ids = set()
answers_checked = 0
errors = []

for article in dataset["data"]:
    if article["title"] not in selection["article_titles"]:
        continue

    for paragraph in article["paragraphs"]:
        context = paragraph["context"]

        for qa in paragraph["qas"]:
            if qa["id"] not in selected_ids:
                continue

            found_ids.add(qa["id"])

            if not qa["question"].strip() or not qa["answers"]:
                raise ValueError(f"Invalid question: {qa['id']}")

            for answer in qa["answers"]:
                start = answer["answer_start"]
                end = start + len(answer["text"])
                actual_text = context[start:end]

                answers_checked += 1

                if (
                    not answer["text"].strip()
                    or start < 0
                    or end > len(context)
                    or actual_text != answer["text"]
                ):
                    errors.append({
                        "question_id": qa["id"],
                        "expected": answer["text"],
                        "actual": actual_text,
                        "start": start,
                    })

missing_ids = sorted(selected_ids - found_ids)

report = {
    "dataset_sha256": dataset_hash,
    "questions_checked": len(found_ids),
    "answers_checked": answers_checked,
    "missing_question_ids": missing_ids,
    "answer_span_errors": errors,
}

Path("data/validation_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("Questions checked:", len(found_ids))
print("Reference answers checked:", answers_checked)
print("Missing questions:", len(missing_ids))
print("Answer span errors:", len(errors))

if missing_ids or errors:
    raise ValueError("Data validation failed. See validation_report.json")

print("Data validation passed.")