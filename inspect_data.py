import json
from pathlib import Path

dataset = json.loads(
    Path("data/dev-v1.1.json").read_text(encoding="utf-8")
)

article = next(
    item for item in dataset["data"]
    if item["title"] == "Warsaw"
)

paragraph = article["paragraphs"][0]
context = paragraph["context"]
qa = paragraph["qas"][0]
answer = qa["answers"][0]

start = answer["answer_start"]
end = start + len(answer["text"])

print("Article:", article["title"])
print("Question:", qa["question"])
print("Reference answer:", answer["text"])
print("Answer starts at:", start)
print("Text at this position:", context[start:end])
print("Match:", context[start:end] == answer["text"])