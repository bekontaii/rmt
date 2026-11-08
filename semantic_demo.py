import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MODEL_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
BREAKPOINT_PERCENTILE = 90

sentences = [
    "Cats and dogs are common household pets.",
    "Many people keep cats and dogs at home.",
    "Python is a programming language.",
    "Developers use Python to write computer programs.",
]

print("Loading model...")

model = SentenceTransformer(
    MODEL_NAME,
    revision=MODEL_REVISION,
    device="cpu",
)

vectors = model.encode(
    sentences,
    normalize_embeddings=True,
    show_progress_bar=False,
)

print("Embedding shape:", vectors.shape)

distances = []

for i in range(len(sentences) - 1):
    similarity = float(np.sum(vectors[i] * vectors[i + 1]))
    distance = 1.0 - similarity
    distances.append(distance)

    print(
        f"Sentences {i + 1}-{i + 2}: "
        f"similarity={similarity:.3f}, "
        f"distance={distance:.3f}"
    )

threshold = float(
    np.percentile(distances, BREAKPOINT_PERCENTILE)
)

print(f"\nThreshold: {threshold:.3f}")

chunks = []
current_sentences = [sentences[0]]

for i, distance in enumerate(distances):
    if distance > threshold:
        chunks.append(" ".join(current_sentences))
        current_sentences = []

    current_sentences.append(sentences[i + 1])

chunks.append(" ".join(current_sentences))

print("\nSemantic chunks:", len(chunks))

for number, chunk in enumerate(chunks, start=1):
    print(f"{number}. {chunk}")