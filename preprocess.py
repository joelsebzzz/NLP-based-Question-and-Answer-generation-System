from transformers import AutoTokenizer

def get_tokenizer(model_checkpoint):
    """Loads tokenizer for the given model checkpoint."""
    print(f"\nLoading tokenizer for {model_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return tokenizer

def preprocess_qna_data(examples, tokenizer, input_prefix, output_structure, max_input_len, max_target_len, qna_type="Short"):
    """Prepares Q&A data for T5 fine-tuning."""
    inputs = []
    targets = []

    contexts = examples.get('context', [])
    questions = examples.get('question', [])
    answers = examples.get('answer', [])

    if not (len(contexts) == len(questions) == len(answers)):
        print(f"Warning ({qna_type} Q&A): Mismatch in lengths: contexts ({len(contexts)}), questions ({len(questions)}), answers ({len(answers)})")
        min_len = min(len(contexts), len(questions), len(answers))
        contexts, questions, answers = contexts[:min_len], questions[:min_len], answers[:min_len]

    for context, question, answer in zip(contexts, questions, answers):
        if not all(isinstance(item, str) for item in [context, question, answer]):
            print(f"Warning ({qna_type} Q&A): Skipping record due to non-string data: Context type {type(context)}, Q type {type(question)}, A type {type(answer)}")
            continue

        model_input_text = f"{input_prefix}{context.strip()}"
        inputs.append(model_input_text)

        model_target_text = output_structure.format(question.strip(), answer.strip())
        targets.append(model_target_text)

    model_inputs = tokenizer(inputs,
                             max_length=max_input_len,
                             padding="max_length",
                             truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets,
                           max_length=max_target_len,
                           padding="max_length",
                           truncation=True)

    label_pad_token_id = -100
    padded_labels = []
    for label_ids in labels["input_ids"]:
        padded_labels.append([
            (l if l != tokenizer.pad_token_id else label_pad_token_id) for l in label_ids
        ])
    model_inputs["labels"] = padded_labels
    return model_inputs

def tokenize_datasets(raw_datasets, tokenizer, input_prefix, output_structure, max_input_len, max_target_len, qna_type="Short"):
    print(f"\nApplying preprocessing to the {qna_type} Q&A datasets...")
    tokenized_ds = raw_datasets.map(
        lambda examples: preprocess_qna_data(examples, tokenizer, input_prefix, output_structure, max_input_len, max_target_len, qna_type),
        batched=True,
        remove_columns=raw_datasets["train"].column_names  # remove original columns
    )
    print(f"Preprocessing finished for {qna_type} Q&A.")

    if len(tokenized_ds['train']) > 0:
        print(f"\nSample Processed {qna_type} Q&A Input (decoded):")
        print(tokenizer.decode(tokenized_ds['train'][0]['input_ids'], skip_special_tokens=False))
        print(f"\nSample Processed {qna_type} Q&A Label (decoded):")
        label_ids_inspect = [id_val for id_val in tokenized_ds['train'][0]['labels'] if id_val != -100]
        print(tokenizer.decode(label_ids_inspect, skip_special_tokens=False))
    else:
        print(f"Tokenized {qna_type} Q&A training dataset is empty, skipping sample inspection.")
    return tokenized_ds