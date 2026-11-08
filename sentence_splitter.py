from nltk.tokenize.punkt import PunktTokenizer


def split_sentences(article, text, sentence_tokenizer):
    sentences = []

    for start, end in sentence_tokenizer.span_tokenize(text):
        sentences.append({
            "article": article,
            "start": start,
            "end": end,
            "text": text[start:end],
        })

    return sentences


if __name__ == "__main__":
    sentence_tokenizer = PunktTokenizer("english")

    demo = (
        'Dr. Smith studies animals. '
        '"Cats are common pets." '
        'Python is a programming language.'
    )

    sentences = split_sentences(
        "demo_article",
        demo,
        sentence_tokenizer,
    )

    print("Sentences:", len(sentences))

    for number, sentence in enumerate(sentences, start=1):
        print(
            f"{number}. [{sentence['start']}:{sentence['end']}] "
            f"{sentence['text']}"
        )