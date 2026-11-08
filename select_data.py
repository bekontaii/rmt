import hashlib
import json
import random
from pathlib import Path

SEED = 42
N_ARTICLES = 20

EXCLUDED_ARTICLES = {
    "Warsaw",
    "Teacher",
    "Martin_Luther",
    "Southern_California",
    "Sky_(United_Kingdom)",
    "Apollo_program",
    "European_Union_law",
    "Ctenophora",
    "Doctor_Who",
    "Prime_number",
}

# 1. Читаем датасет
dataset = json.loads(
    Path("data/dev-v1.1.json").read_text(encoding="utf-8")
)

# 2. Создаём список подходящих статей
eligible_articles = []
excluded_count = 0

for article in dataset["data"]:
    if article["title"] in EXCLUDED_ARTICLES:
        excluded_count += 1
        continue

    question_count = 0

    for paragraph in article["paragraphs"]:
        question_count += len(paragraph["qas"])

    if question_count >= 100:
        eligible_articles.append(article)

print("All articles:", len(dataset["data"]))
print("Excluded A3 articles:", excluded_count)
print("Eligible articles:", len(eligible_articles))

# 3. Выбираем 20 статей из подготовленного списка
rng = random.Random(SEED)

selected_articles = rng.sample(
    eligible_articles,
    N_ARTICLES,
)

print("\nSelected articles:")

for number, article in enumerate(selected_articles, start=1):
    print(f"{number}. {article['title']}")

print("Selected count:", len(selected_articles))

retrieval_ids = []
generation_ids = []

for article in selected_articles:
    article_questions = []

    for paragraph in article["paragraphs"]:
        article_questions.extend(paragraph["qas"])

    sampled_questions = rng.sample(article_questions, 100)

    for qa in sampled_questions:
        retrieval_ids.append(qa["id"])

    for qa in sampled_questions[:50]:
        generation_ids.append(qa["id"])

# Проверяем количество и отсутствие повторений
assert len(retrieval_ids) == 2000
assert len(generation_ids) == 1000
assert len(set(retrieval_ids)) == 2000
assert len(set(generation_ids)) == 1000
assert set(generation_ids).issubset(set(retrieval_ids))

manifest = {
    "seed": SEED,
    "dataset_sha256": hashlib.sha256(
        Path("data/dev-v1.1.json").read_bytes()
    ).hexdigest(),
    "excluded_a3_articles": sorted(EXCLUDED_ARTICLES),
    "article_titles": [
        article["title"] for article in selected_articles
    ],
    "retrieval_questions_per_article": 100,
    "generation_questions_per_article": 50,
    "retrieval_question_ids": retrieval_ids,
    "generation_question_ids": generation_ids,
}

output_path = Path("data/selection_manifest.json")

output_path.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\nRetrieval questions:", len(retrieval_ids))
print("Generation questions:", len(generation_ids))
print("Selection saved:", output_path)