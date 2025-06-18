import os
import torch
import nltk
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

import config
from data_loader import load_qna_datasets
from preprocess import get_tokenizer, tokenize_datasets
from model_utils import compute_qna_metrics


def main():
    # NLTK setup
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')  # For metrics function
    except nltk.downloader.DownloadError:
        print("Downloading NLTK punkt_tab for metrics...")
        nltk.download('punkt_tab', quiet=True)

    config.print_long_qna_config()

    # 1. Load Tokenizer
    tokenizer = get_tokenizer(config.MODEL_CHECKPOINT_LONG)

    # 2. Load Raw Datasets
    raw_datasets_long = load_qna_datasets(
        config.LONG_QNA_TRAIN_PATH,
        config.LONG_QNA_VAL_PATH,
        qna_type="Long"
    )

    # 3. Preprocess and Tokenize Datasets
    if not raw_datasets_long["train"] or not raw_datasets_long["validation"]:
        print("Long Q&A training or validation dataset is empty. Exiting training.")
        return

    tokenized_datasets_long = tokenize_datasets(
        raw_datasets_long,
        tokenizer,
        config.INPUT_PREFIX_LONG,
        config.OUTPUT_STRUCTURE_LONG,
        config.MAX_INPUT_LENGTH_LONG,
        config.MAX_TARGET_LENGTH_LONG,
        qna_type="Long"
    )

    # 4. Load Base Model
    print(f"\nLoading base model '{config.MODEL_CHECKPOINT_LONG}' for Long Q&A training...")
    model_long = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_CHECKPOINT_LONG)
    model_long.to(config.DEVICE)
    print("Base model for Long Q&A loaded and moved to device.")

    # 5. Training Arguments
    print(f"\nSetting up Training Arguments for Long Q&A. Output dir: {config.LONG_QNA_OUTPUT_DIR}")
    # Similar calculation for save_steps if needed, or rely on save_strategy="epoch"

    args_long = Seq2SeqTrainingArguments(
        output_dir=config.LONG_QNA_OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=config.LEARNING_RATE_LONG,
        per_device_train_batch_size=config.BATCH_SIZE_LONG,
        per_device_eval_batch_size=config.BATCH_SIZE_LONG * 2,
        weight_decay=config.WEIGHT_DECAY_LONG,
        save_total_limit=3,
        num_train_epochs=config.NUM_TRAIN_EPOCHS_LONG,
        predict_with_generate=True,
        fp16=config.FP16_TRAINING,
        logging_dir=os.path.join(config.LONG_QNA_OUTPUT_DIR, "logs_long"),  # Differentiated log dir
        logging_strategy="steps",
        logging_steps=config.LOGGING_STEPS_LONG,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=["tensorboard"],
        generation_max_length=config.MAX_TARGET_LENGTH_LONG
    )

    # 6. Data Collator
    print("Setting up Data Collator for Long Q&A...")
    data_collator_long = DataCollatorForSeq2Seq(tokenizer, model=model_long)
    print("Setup complete for Long Q&A.")

    # 7. Trainer
    print("\nSetting up Trainer for Long Q&A task...")
    trainer_long = Seq2SeqTrainer(
        model=model_long,
        args=args_long,
        train_dataset=tokenized_datasets_long["train"],
        eval_dataset=tokenized_datasets_long["validation"],
        data_collator=data_collator_long,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_qna_metrics(p, tokenizer)
    )
    print("Trainer for Long Q&A setup complete.")

    # 8. Train
    print("\nStarting training for Long Q&A Generation...")
    try:
        train_result_long = trainer_long.train()
        print("Long Q&A training finished.")

        trainer_long.save_model()
        print(f"Final Long Q&A model (best model) saved to {config.LONG_QNA_OUTPUT_DIR}")

        metrics_long = train_result_long.metrics
        trainer_long.log_metrics("train_long", metrics_long)
        trainer_long.save_metrics("train_long", metrics_long)
        trainer_long.save_state()

        long_qna_best_model_path = os.path.join(config.LONG_QNA_OUTPUT_DIR, "best_model")
        if not os.path.exists(long_qna_best_model_path):
            os.makedirs(long_qna_best_model_path)
        # trainer_long.save_model(long_qna_best_model_path) # Redundant if load_best_model_at_end=True
        print(f"Best Long Q&A Model explicitly available at {long_qna_best_model_path}")

    except Exception as e:
        print(f"An error occurred during Long Q&A training: {e}")
        if torch.cuda.is_available():
            print("Attempting to clear CUDA cache for Long Q&A training...")
            torch.cuda.empty_cache()
        raise

    finally:
        if torch.cuda.is_available():
            print("Cleaning up Long Q&A model and trainer from GPU memory (if applicable)...")
            if 'model_long' in locals(): del model_long
            if 'trainer_long' in locals(): del trainer_long
            torch.cuda.empty_cache()

    print("\nTraining process complete for Long Q&A.")


if __name__ == "__main__":
    main()