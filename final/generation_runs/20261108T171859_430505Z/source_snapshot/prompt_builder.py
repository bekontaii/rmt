import json
import re
from copy import deepcopy
from pathlib import Path

from transformers import AutoTokenizer

from retrieval_metrics import validate_span


def count_tokens(text, tokenizer, limit):
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        max_length=limit,
    )

    return len(encoded["input_ids"])


def build_prompt(question, retrieved_results, tokenizer, config):
    generation = config["generation"]

    limit = generation["max_input_tokens"]
    top_k = generation["context_top_k"]
    template = generation["prompt_template"]

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a nonempty string")

    if type(limit) is not int or limit <= 0:
        raise ValueError("Input token limit must be positive")

    if type(top_k) is not int or top_k <= 0:
        raise ValueError("context_top_k must be positive")

    if generation["context_policy"] != "ranked_prefix_whole_words":
        raise ValueError("Unsupported context policy")

    if "{question}" not in template or "{context}" not in template:
        raise ValueError(
            "Prompt template must contain {question} and {context}"
        )

    if len(retrieved_results) < top_k:
        raise ValueError(
            f"Expected at least {top_k} retrieved results"
        )

    selected_chunks = [
        result["chunk"]
        for result in retrieved_results[:top_k]
    ]

    for chunk in selected_chunks:
        text = chunk["text"]
        start, end = validate_span([
            chunk["start"],
            chunk["end"],
        ])

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Chunk text must be nonempty")

        if end - start != len(text):
            raise ValueError("Chunk text length does not match offsets")

    # Сначала проверяем, помещается ли вопрос без контекста.
    minimum_prompt = template.format(
        context="",
        question=question,
    )

    minimum_tokens = count_tokens(
        minimum_prompt,
        tokenizer,
        limit,
    )

    if minimum_tokens > limit:
        raise ValueError(
            "The full question and instruction exceed "
            f"the input limit: {minimum_tokens} > {limit}"
        )

    full_context = "\n\n".join(
        chunk["text"]
        for chunk in selected_chunks
    )

    word_ends = [
        match.end()
        for match in re.finditer(r"\S+", full_context)
    ]

    context = full_context
    kept_words = len(word_ends)

    prompt = template.format(
        context=context,
        question=question,
    )

    input_tokens = count_tokens(
        prompt,
        tokenizer,
        limit,
    )

    full_input_tokens = input_tokens

    # Удаляем последние слова контекста до попадания в лимит.
    while input_tokens > limit:
        if kept_words == 0:
            raise RuntimeError("Unable to fit prompt into the limit")

        kept_words -= 1

        if kept_words:
            context = full_context[:word_ends[kept_words - 1]]
        else:
            context = ""

        prompt = template.format(
            context=context,
            question=question,
        )

        input_tokens = count_tokens(
            prompt,
            tokenizer,
            limit,
        )

    # Сохраняем исходные позиции реально видимых фрагментов.
    visible_chunks = []
    context_offset = 0

    for rank, chunk in enumerate(selected_chunks, start=1):
        used_characters = max(
            0,
            min(
                len(chunk["text"]),
                len(context) - context_offset,
            ),
        )

        if used_characters > 0:
            visible = dict(chunk)

            visible["rank"] = rank
            visible["original_end"] = chunk["end"]
            visible["end"] = chunk["start"] + used_characters
            visible["text"] = chunk["text"][:used_characters]

            visible_chunks.append(visible)

        context_offset += len(chunk["text"]) + 2

    return {
        "prompt": prompt,
        "context": context,
        "input_tokens": input_tokens,
        "full_input_tokens": full_input_tokens,
        "context_truncated": context != full_context,
        "context_words_full": len(word_ends),
        "context_words_used": kept_words,
        "visible_chunks": visible_chunks,
    }


if __name__ == "__main__":
    project_directory = Path(__file__).resolve().parent

    config = json.loads(
        (project_directory / "experiment_config.json").read_text(
            encoding="utf-8"
        )
    )

    saved_files = sorted(
        (
            project_directory
            / "development"
            / "toy_runs"
        ).glob("rag_demo_*.json")
    )

    if not saved_files:
        raise FileNotFoundError(
            "No saved toy results. Run python rag_demo.py first."
        )

    record = json.loads(
        saved_files[-1].read_text(encoding="utf-8")
    )

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        config["generation"]["name"],
        revision=config["generation"]["revision"],
        use_fast=False,
        legacy=True,
    )

    question = record["question"]
    retrieved_results = record["retrieved_results"]

    # Проверка 1: обычный вход должен сохраниться полностью.
    normal = build_prompt(
        question,
        retrieved_results,
        tokenizer,
        config,
    )

    assert normal["prompt"] == record["prompt"]
    assert normal["input_tokens"] == record["input_tokens"]
    assert not normal["context_truncated"]

    print("\nNormal input tokens:", normal["input_tokens"])
    print("Normal context truncated:", normal["context_truncated"])

    # Проверка 2: уменьшаем лимит в отдельной копии настроек.
    test_config = deepcopy(config)

    test_config["generation"]["max_input_tokens"] = (
        normal["input_tokens"] - 1
    )

    shortened = build_prompt(
        question,
        retrieved_results,
        tokenizer,
        test_config,
    )

    test_limit = test_config["generation"]["max_input_tokens"]

    assert shortened["input_tokens"] <= test_limit
    assert shortened["context_truncated"]
    assert question in shortened["prompt"]

    assert normal["context"].startswith(shortened["context"])

    # Проверяем позиции текста после сокращения.
    for packed in (normal, shortened):
        for chunk in packed["visible_chunks"]:
            source = record["documents"][chunk["article"]]

            assert source[
                chunk["start"]:chunk["end"]
            ] == chunk["text"]

    print("\nReduced test limit:", test_limit)
    print("Shortened input tokens:", shortened["input_tokens"])
    print("Context truncated:", shortened["context_truncated"])
    print("Full question preserved:", question in shortened["prompt"])
    print("Visible chunk offsets: OK")

    # Проверка 3: слишком маленький лимит должен вызвать ошибку.
    tiny_config = deepcopy(config)
    tiny_config["generation"]["max_input_tokens"] = 1

    try:
        build_prompt(
            question,
            retrieved_results,
            tokenizer,
            tiny_config,
        )
    except ValueError:
        print("Too-small budget check: OK")
    else:
        raise AssertionError("Expected an input budget error")

    print("\nPrompt builder checks passed.")