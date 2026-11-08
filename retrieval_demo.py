import numpy as np
from sentence_transformers import SentenceTransformer

from chunkers import fixed_chunks
from semantic_chunker import MODEL_NAME, MODEL_REVISION


def retrieve_top_k(
    question,
    chunks,
    chunk_vectors,
    model,
    top_k=3,
):
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if not chunks:
        raise ValueError("No chunks to search")

    if chunk_vectors.ndim != 2:
        raise ValueError("Chunk embeddings must be a matrix")

    if chunk_vectors.shape[0] != len(chunks):
        raise ValueError("Chunk and embedding counts do not match")

    # Вопрос кодируется той же моделью, что и чанки.
    question_vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    if question_vector.shape != (chunk_vectors.shape[1],):
        raise ValueError("Embedding dimensions do not match")

    if not (
        np.isfinite(chunk_vectors).all()
        and np.isfinite(question_vector).all()
    ):
        raise ValueError("Embeddings contain invalid values")

    # Для нормализованных векторов это cosine similarity.
    scores = np.sum(
        chunk_vectors * question_vector,
        axis=1,
    )

    if not np.isfinite(scores).all():
        raise ValueError("Similarity scores contain invalid values")

    # Сортируем от наибольшего сходства к наименьшему.
    ranking = np.argsort(
        -scores,
        kind="stable",
    )[:top_k]

    results = []

    for index in ranking:
        index = int(index)

        results.append({
            "chunk": chunks[index],
            "score": float(scores[index]),
        })

    return results


if __name__ == "__main__":
    documents = {
        "Pets": (
            "Cats and dogs are common household pets."
        ),
        "Programming": (
            "Python is a programming language. "
            "Developers use Python to write computer programs."
        ),
        "Geography": (
            "Paris is the capital city of France."
        ),
        "Plants": (
            "Plants use sunlight to produce energy "
            "through photosynthesis."
        ),
    }

    chunks = []

    for article, text in documents.items():
        chunks.extend(
            fixed_chunks(
                article,
                text,
                size=100,
                overlap=0,
            )
        )

    print("Loading model...")

    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device="cpu",
    )

    # Представления чанков вычисляются один раз.
    chunk_vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    question = (
        "Which programming language do developers "
        "use to write computer programs?"
    )

    results = retrieve_top_k(
        question,
        chunks,
        chunk_vectors,
        model,
        top_k=3,
    )

    print("\nChunks indexed:", len(chunks))
    print("Embedding shape:", chunk_vectors.shape)
    print("Question:", question)
    print("\nTop 3 retrieved chunks:")

    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print(
            f"\n{rank}. Article: {chunk['article']} | "
            f"Score: {result['score']:.4f}"
        )
        print(chunk["text"])