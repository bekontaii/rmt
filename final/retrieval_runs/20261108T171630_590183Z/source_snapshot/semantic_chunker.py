import numpy as np
from nltk.tokenize.punkt import PunktTokenizer
from sentence_transformers import SentenceTransformer

from chunkers import fixed_chunks
from sentence_splitter import split_sentences


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MODEL_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"


def prepare_units(article, text, sentence_tokenizer, max_words=100):
    if max_words <= 0:
        raise ValueError("Maximum words must be positive")

    sentences = split_sentences(
        article,
        text,
        sentence_tokenizer,
    )

    units = []

    for sentence in sentences:
        pieces = fixed_chunks(
            article,
            sentence["text"],
            size=max_words,
            overlap=0,
        )

        for piece in pieces:
            start = sentence["start"] + piece["start"]
            end = sentence["start"] + piece["end"]

            units.append({
                "article": article,
                "start": start,
                "end": end,
                "text": text[start:end],
            })

    return units


def semantic_chunks(
    article,
    text,
    model,
    sentence_tokenizer,
    max_words=100,
    breakpoint_percentile=90,
):
    if not 0 <= breakpoint_percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")

    units = prepare_units(
        article,
        text,
        sentence_tokenizer,
        max_words=max_words,
    )

    if len(units) <= 1:
        return units

    unit_texts = [unit["text"] for unit in units]

    # Проверяем, что модель сможет обработать каждый фрагмент целиком.
    tokenized = model.tokenizer(
        unit_texts,
        add_special_tokens=True,
        truncation=False,
    )

    longest = max(
        len(ids)
        for ids in tokenized["input_ids"]
    )

    if longest > model.max_seq_length:
        raise ValueError(
            f"Unit has {longest} tokens; "
            f"model limit is {model.max_seq_length}"
        )

    vectors = model.encode(
        unit_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain invalid values")

    # Расстояния между соседними фрагментами.
    distances = []

    for i in range(len(units) - 1):
        similarity = float(
            np.sum(vectors[i] * vectors[i + 1])
        )

        distances.append(1.0 - similarity)

    threshold = float(
        np.percentile(distances, breakpoint_percentile)
    )

    # Собираем чанки с учётом смысла и ограничения длины.
    chunks = []

    start = units[0]["start"]
    end = units[0]["end"]

    for i in range(1, len(units)):
        unit = units[i]

        candidate_text = text[start:unit["end"]]

        semantic_break = distances[i - 1] > threshold
        length_break = len(candidate_text.split()) > max_words

        if semantic_break or length_break:
            chunks.append({
                "article": article,
                "start": start,
                "end": end,
                "text": text[start:end],
            })

            start = unit["start"]

        end = unit["end"]

    chunks.append({
        "article": article,
        "start": start,
        "end": end,
        "text": text[start:end],
    })

    return chunks


if __name__ == "__main__":
    sentence_tokenizer = PunktTokenizer("english")

    print("Loading model...")

    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device="cpu",
    )

    # Проверка 1: две разные темы.
    demo = (
        "Cats and dogs are common household pets. "
        "Many people keep cats and dogs at home. "
        "Python is a programming language. "
        "Developers use Python to write computer programs."
    )

    chunks = semantic_chunks(
        "demo_article",
        demo,
        model,
        sentence_tokenizer,
    )

    print("\nSemantic chunks:", len(chunks))

    for number, chunk in enumerate(chunks, start=1):
        start = chunk["start"]
        end = chunk["end"]

        assert demo[start:end] == chunk["text"]
        assert len(chunk["text"].split()) <= 100

        print(
            f"{number}. [{start}:{end}] "
            f"{chunk['text']}"
        )

    recovered_words = [
        word
        for chunk in chunks
        for word in chunk["text"].split()
    ]

    assert recovered_words == demo.split()

    print("\nOffsets, length and coverage: OK")

    # Проверка 2: одно предложение из 251 слова.
    long_demo = " ".join(
        f"word{i}" for i in range(1, 252)
    ) + "."

    long_chunks = semantic_chunks(
        "long_demo_article",
        long_demo,
        model,
        sentence_tokenizer,
        max_words=100,
    )

    sizes = [
        len(chunk["text"].split())
        for chunk in long_chunks
    ]

    assert sizes == [100, 100, 51]

    for chunk in long_chunks:
        start = chunk["start"]
        end = chunk["end"]

        assert long_demo[start:end] == chunk["text"]

    recovered_words = [
        word
        for chunk in long_chunks
        for word in chunk["text"].split()
    ]

    assert recovered_words == long_demo.split()

    for previous, following in zip(
        long_chunks,
        long_chunks[1:],
    ):
        assert previous["end"] <= following["start"]

    print("\nLong sentence words:", len(long_demo.split()))
    print("Semantic chunks:", len(long_chunks))
    print("Chunk sizes:", sizes)
    print("Length cap, offsets and coverage: OK")