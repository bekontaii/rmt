import re


def fixed_chunks(article, text, size=100, overlap=0):
    if size <= 0:
        raise ValueError("Chunk size must be positive")

    if not 0 <= overlap < size:
        raise ValueError("Overlap must be between 0 and size - 1")

    words = list(re.finditer(r"\S+", text))
    chunks = []

    step = size - overlap

    for i in range(0, len(words), step):
        window = words[i:i + size]

        start = window[0].start()
        end = window[-1].end()

        chunks.append({
            "article": article,
            "start": start,
            "end": end,
            "text": text[start:end],
        })

        if i + size >= len(words):
            break

    return chunks


if __name__ == "__main__":
    demo = "one two three four five six seven eight nine ten eleven twelve"

    for name, overlap in [("Fixed", 0), ("Overlap", 2)]:
        chunks = fixed_chunks(
            "demo_article",
            demo,
            size=5,
            overlap=overlap,
        )

        print(f"\n{name}: {len(chunks)} chunks")

        for number, chunk in enumerate(chunks, start=1):
            print(
                f"{number}. [{chunk['start']}:{chunk['end']}] "
                f"{chunk['text']}"
            )