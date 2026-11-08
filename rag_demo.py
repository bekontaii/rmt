import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from chunkers import fixed_chunks
from metrics import score_answer
from retrieval_demo import retrieve_top_k

from semantic_chunker import (
    MODEL_NAME as EMBEDDING_NAME,
    MODEL_REVISION as EMBEDDING_REVISION,
)

from generation_demo import (
    MODEL_NAME as GENERATOR_NAME,
    MODEL_REVISION as GENERATOR_REVISION,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
)


def generate_answer(question, retrieved_results, tokenizer, model):
    # Собираем контекст из найденных фрагментов.
    context = "\n\n".join(
        result["chunk"]["text"]
        for result in retrieved_results
    )

    prompt = (
        "Answer the question using only the context. "
        "Give a short answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False,
    )

    input_tokens = inputs["input_ids"].shape[1]

    if input_tokens > MAX_INPUT_TOKENS:
        raise ValueError(
            f"Input has {input_tokens} tokens; "
            f"experiment limit is {MAX_INPUT_TOKENS}"
        )

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
        )

    prediction = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    ).strip()

    return {
        "prediction": prediction,
        "input_tokens": input_tokens,
        "prompt": prompt,
    }


if __name__ == "__main__":
    # Учебные документы.
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

    question = (
        "Which programming language do developers "
        "use to write computer programs?"
    )

    # Создаём чанки.
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

    # Загружаем модель поиска.
    print("Loading embedding model:", EMBEDDING_NAME)

    embedding_model = SentenceTransformer(
        EMBEDDING_NAME,
        revision=EMBEDDING_REVISION,
        device="cpu",
    )

    chunk_vectors = embedding_model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    # Находим три наиболее близких фрагмента.
    retrieved_results = retrieve_top_k(
        question,
        chunks,
        chunk_vectors,
        embedding_model,
        top_k=3,
    )

    print("\nQuestion:", question)
    print("\nRetrieved chunks:")

    for rank, result in enumerate(retrieved_results, start=1):
        chunk = result["chunk"]

        print(
            f"{rank}. {chunk['article']} | "
            f"Score: {result['score']:.4f}"
        )
        print(chunk["text"])

    # Освобождаем модель поиска перед загрузкой генератора.
    del embedding_model
    gc.collect()

    # Загружаем генератор.
    print("\nLoading generator:", GENERATOR_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        GENERATOR_NAME,
        revision=GENERATOR_REVISION,
        use_fast=False,
        legacy=True,
    )

    generator = AutoModelForSeq2SeqLM.from_pretrained(
        GENERATOR_NAME,
        revision=GENERATOR_REVISION,
        dtype=torch.float32,
    )

    generator.to("cpu")
    generator.eval()

    # Генерируем ответ по найденному контексту.
    print("Generating answer...")

    answer = generate_answer(
        question,
        retrieved_results,
        tokenizer,
        generator,
    )

    # Эталон используется только для оценки.
    reference_answer = "Python"

    scores = score_answer(
        answer["prediction"],
        [reference_answer],
    )

    print("\nInput tokens:", answer["input_tokens"])
    print("\nPrompt sent to generator:\n")
    print(answer["prompt"])

    print("\nGenerated answer:", answer["prediction"])
    print("Reference answer:", reference_answer)

    print(f"\nExact Match: {scores['exact_match']:.3f}")
    print(f"Token F1: {scores['f1']:.3f}")

    # Подготавливаем запись результата.
    recorded_at = datetime.now(timezone.utc)

    record = {
        "run_type": "toy_development_demo",
        "recorded_at_utc": recorded_at.isoformat(),
        "models": {
            "embedding": {
                "name": EMBEDDING_NAME,
                "revision": EMBEDDING_REVISION,
            },
            "generator": {
                "name": GENERATOR_NAME,
                "revision": GENERATOR_REVISION,
            },
        },
        "documents": documents,
        "indexed_chunks": chunks,
        "question": question,
        "reference_answers": [reference_answer],
        "retrieved_results": retrieved_results,
        "prompt": answer["prompt"],
        "input_tokens": answer["input_tokens"],
        "prediction": answer["prediction"],
        "metrics": scores,
    }

    # Создаём папку для учебных запусков.
    output_directory = (
        Path(__file__).resolve().parent
        / "development"
        / "toy_runs"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Каждый запуск сохраняется отдельно.
    timestamp = recorded_at.strftime("%Y%m%dT%H%M%S_%fZ")

    output_path = (
        output_directory
        / f"rag_demo_{timestamp}.json"
    )

    with output_path.open("x", encoding="utf-8") as output_file:
        json.dump(
            record,
            output_file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        output_file.write("\n")

    print("\nSaved:", output_path)