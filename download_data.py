from pathlib import Path
from urllib.request import urlretrieve
import hashlib
import json

URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
FILE_PATH = Path("data/dev-v1.1.json")

EXPECTED_SHA256 = (
    "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9"
)

FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

if not FILE_PATH.exists():
    print("Downloading SQuAD 1.1...")
    urlretrieve(URL, FILE_PATH)

raw_data = FILE_PATH.read_bytes()
file_hash = hashlib.sha256(raw_data).hexdigest()

if file_hash != EXPECTED_SHA256:
    raise ValueError("Dataset checksum does not match")

dataset = json.loads(raw_data)
articles = dataset["data"]

question_count = 0

for article in articles:
    for paragraph in article["paragraphs"]:
        question_count += len(paragraph["qas"])

print("File:", FILE_PATH)
print("Version:", dataset["version"])
print("Articles:", len(articles))
print("Questions:", question_count)
print("SHA256:", file_hash)
