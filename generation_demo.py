import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "google/flan-t5-large"
MODEL_REVISION = "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"

MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 32


if __name__ == "__main__":
    question = (
        "Which programming language do developers "
        "use to write computer programs?"
    )

    # Три текста из предыдущего примера поиска.
    retrieved_texts = [
        (
            "Python is a programming language. "
            "Developers use Python to write computer programs."
        ),
        (
            "Plants use sunlight to produce energy "
            "through photosynthesis."
        ),
        (
            "Cats and dogs are common household pets."
        ),
    ]

    context = "\n\n".join(retrieved_texts)

    prompt = (
        "Answer the question using only the context. "
        "Give a short answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        use_fast=False,
        legacy=True,
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

    print("Loading generator:", MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        dtype=torch.float32,
    )

    model.to("cpu")
    model.eval()

    print("Device:", next(model.parameters()).device)
    print("Input tokens:", input_tokens)
    print("\nPrompt sent to model:\n")
    print(prompt)
    print("\nGenerating answer...")

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

    # Эталон выводится для проверки и не передаётся модели.
    reference_answer = "Python"

    print("\nQuestion:", question)
    print("Generated answer:", prediction)
    print("Reference answer:", reference_answer)