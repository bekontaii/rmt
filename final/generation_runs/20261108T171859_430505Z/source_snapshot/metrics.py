import math
import re
import string
from collections import Counter


PUNCTUATION_TABLE = str.maketrans(
    "",
    "",
    string.punctuation,
)


def normalize_answer(text):
    text = text.lower()
    text = text.translate(PUNCTUATION_TABLE)
    text = re.sub(r"\b(?:a|an|the)\b", " ", text)

    return " ".join(text.split())


def exact_match_score(prediction, reference):
    return float(
        normalize_answer(prediction)
        == normalize_answer(reference)
    )


def f1_score(prediction, reference):
    prediction_words = normalize_answer(prediction).split()
    reference_words = normalize_answer(reference).split()

    matching_words = (
        Counter(prediction_words)
        & Counter(reference_words)
    )

    matched_count = sum(matching_words.values())

    if matched_count == 0:
        return 0.0

    return (
        2.0 * matched_count
        / (len(prediction_words) + len(reference_words))
    )


def score_answer(prediction, references):
    if isinstance(references, str):
        raise TypeError("References must be a list of strings")

    if not references:
        raise ValueError("At least one reference is required")

    return {
        "exact_match": max(
            exact_match_score(prediction, reference)
            for reference in references
        ),
        "f1": max(
            f1_score(prediction, reference)
            for reference in references
        ),
    }


if __name__ == "__main__":
    cases = [
        (
            "Exact answer",
            "Python",
            ["Python"],
            1.0,
            1.0,
        ),
        (
            "Normalized answer",
            "The Nobel Prize!",
            ["Nobel Prize"],
            1.0,
            1.0,
        ),
        (
            "Partial answer",
            "Lutheran hymnals",
            ["early Lutheran hymnals"],
            0.0,
            0.8,
        ),
        (
            "Wrong answer",
            "Justus Jonas",
            ["Eisleben"],
            0.0,
            0.0,
        ),
        (
            "Multiple references",
            "Python",
            ["Python programming language", "Python"],
            1.0,
            1.0,
        ),
        (
            "Repeated words",
            "Paris Paris",
            ["Paris"],
            0.0,
            2.0 / 3.0,
        ),
    ]

    for name, prediction, references, expected_em, expected_f1 in cases:
        scores = score_answer(prediction, references)

        assert scores["exact_match"] == expected_em
        assert math.isclose(
            scores["f1"],
            expected_f1,
            abs_tol=1e-12,
        )

        print(
            f"{name}: "
            f"EM={scores['exact_match']:.3f}, "
            f"F1={scores['f1']:.3f}"
        )

    # Граничные случаи по правилам SQuAD 1.1.
    assert score_answer("", ["Python"]) == {
        "exact_match": 0.0,
        "f1": 0.0,
    }

    assert score_answer("", [""]) == {
        "exact_match": 1.0,
        "f1": 0.0,
    }

    print("\nAll metric checks passed.")