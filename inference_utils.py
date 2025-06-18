import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
import os
import config  # Import your config file

# --- Short Q&A Inference ---
_tokenizer_inf_short = None
_model_inf_short = None
_device_inf_short = None


def load_short_qna_model_for_inference():
    global _tokenizer_inf_short, _model_inf_short, _device_inf_short
    if _model_inf_short is None:
        if not os.path.exists(config.SHORT_QNA_MODEL_LOAD_PATH):
            raise FileNotFoundError(
                f"Fine-tuned Short Q&A model directory not found: {config.SHORT_QNA_MODEL_LOAD_PATH}.")

        print(f"Loading fine-tuned Short Q&A model and tokenizer from: {config.SHORT_QNA_MODEL_LOAD_PATH}")
        _tokenizer_inf_short = AutoTokenizer.from_pretrained(config.SHORT_QNA_MODEL_LOAD_PATH)
        _model_inf_short = AutoModelForSeq2SeqLM.from_pretrained(config.SHORT_QNA_MODEL_LOAD_PATH)

        _device_inf_short = config.DEVICE
        _model_inf_short.to(_device_inf_short)
        _model_inf_short.eval()
        print(f"Short Q&A Inference model moved to device: {_device_inf_short}")
    return _tokenizer_inf_short, _model_inf_short, _device_inf_short


def parse_qna_output(generated_text):
    """Parses 'question: Q answer: A' format."""
    match = re.match(r"question:\s*(.*?)\s*answer:\s*(.*)", generated_text, re.IGNORECASE | re.DOTALL)
    if match:
        question = match.group(1).strip()
        answer = match.group(2).strip()
        if question and answer:
            # print("Parsing successful.")
            return {"question": question, "answer": answer}

    # Fallback parsing if the primary regex fails
    # print("Primary parsing failed. Attempting fallback parsing...")
    parts = generated_text.lower().split('answer:', 1)
    if len(parts) == 2:
        q_part = parts[0].replace('question:', '').strip()
        a_part = parts[1].strip()
        if q_part and a_part:
            # print("Fallback parsing successful.")
            return {"question": q_part, "answer": a_part}

    # print("Error: Could not parse the generated output into question/answer.")
    return None


def generate_single_short_qna(context):
    tokenizer, model, device = load_short_qna_model_for_inference()
    if not context or not isinstance(context, str):
        print("Error: Invalid context provided for short Q&A.")
        return None

    input_text = f"{config.INPUT_PREFIX_SHORT}{context.strip()}"
    inputs = tokenizer(input_text,
                       max_length=config.MAX_INPUT_LENGTH_SHORT,
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # print(f"\nGenerating short question and answer...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=config.INF_MAX_OUTPUT_LENGTH_SHORT,
            num_beams=config.INF_NUM_BEAMS_SHORT_DEFAULT,
            early_stopping=config.INF_EARLY_STOPPING_SHORT_DEFAULT,
            no_repeat_ngram_size=config.INF_NO_REPEAT_NGRAM_SIZE_SHORT_DEFAULT,
            num_return_sequences=1
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # print(f"Raw generated output (short): {generated_text}")
    return parse_qna_output(generated_text)


# --- Long Q&A Inference ---
_tokenizer_inf_long = None
_model_inf_long = None
_device_inf_long = None


def load_long_qna_model_for_inference():
    global _tokenizer_inf_long, _model_inf_long, _device_inf_long
    if _model_inf_long is None:
        if not os.path.exists(config.LONG_QNA_MODEL_LOAD_PATH):
            raise FileNotFoundError(
                f"Fine-tuned Long Q&A model directory not found: {config.LONG_QNA_MODEL_LOAD_PATH}.")

        print(f"Loading fine-tuned Long Q&A model and tokenizer from: {config.LONG_QNA_MODEL_LOAD_PATH}")
        _tokenizer_inf_long = AutoTokenizer.from_pretrained(config.LONG_QNA_MODEL_LOAD_PATH)
        _model_inf_long = AutoModelForSeq2SeqLM.from_pretrained(config.LONG_QNA_MODEL_LOAD_PATH)

        _device_inf_long = config.DEVICE
        _model_inf_long.to(_device_inf_long)
        _model_inf_long.eval()
        print(f"Long Q&A Inference model moved to device: {_device_inf_long}")
    return _tokenizer_inf_long, _model_inf_long, _device_inf_long


def generate_single_long_qna(context):
    tokenizer, model, device = load_long_qna_model_for_inference()
    if not context or not isinstance(context, str):
        print("Error: Invalid context provided for long Q&A.")
        return None

    input_text = f"{config.INPUT_PREFIX_LONG}{context.strip()}"
    inputs = tokenizer(input_text,
                       max_length=config.MAX_INPUT_LENGTH_LONG,
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # print(f"\nGenerating long question and answer...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=config.INF_MAX_OUTPUT_LENGTH_LONG,
            num_beams=config.INF_NUM_BEAMS_LONG_DEFAULT,
            early_stopping=config.INF_EARLY_STOPPING_LONG_DEFAULT,
            no_repeat_ngram_size=config.INF_NO_REPEAT_NGRAM_SIZE_LONG_DEFAULT,
            num_return_sequences=1
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # print(f"Raw generated LONG output: {generated_text}")
    return parse_qna_output(generated_text)


if __name__ == '__main__':
    # Example usage (optional, for testing this module directly)
    print("--- Example Short Q&A Inference ---")
    passage_example_short = """
    Dynamic typing checks types at runtime. Functional programming emphasizes pure functions and immutability.
    Object-oriented programming organizes code into classes and objects. Static typing enforces type rules at compile time.
    """
    print(f"Input Passage (Short):\n{passage_example_short[:200]}...")
    short_qna_pair = generate_single_short_qna(passage_example_short)
    if short_qna_pair:
        print("\nGenerated Short Q&A Pair:")
        print(f"  Q: {short_qna_pair['question']}")
        print(f"  A: {short_qna_pair['answer']}")
    else:
        print("\nFailed to generate a valid short Q&A pair.")

    print("\n--- Example LONG Q&A Inference ---")
    passage_example_long = """
    A central challenge in Machine Learning is balancing computational efficiency with accuracy.
    Key applications of Machine Learning include real-world problem solving and data analysis.
    Core theoretical concepts in Machine Learning are essential for designing efficient systems.
    Machine Learning often relies on mathematical models and statistical methods for analysis.
    """
    print(f"Input Passage (Long):\n{passage_example_long[:250]}...")
    long_qna_pair = generate_single_long_qna(passage_example_long)
    if long_qna_pair:
        print("\nGenerated LONG Q&A Pair:")
        print(f"  Q: {long_qna_pair['question']}")
        print(f"  A: {long_qna_pair['answer']}")
    else:
        print("\nFailed to generate a valid LONG Q&A pair.")

    # Optional: Clean up GPU memory if models were loaded
    if torch.cuda.is_available():
        if _model_inf_short: del _model_inf_short
        if _tokenizer_inf_short: del _tokenizer_inf_short
        if _model_inf_long: del _model_inf_long
        if _tokenizer_inf_long: del _tokenizer_inf_long
        torch.cuda.empty_cache()
        print("\nCleaned up inference models from memory.")