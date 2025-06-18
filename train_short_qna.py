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

    config.print_short_qna_config()

    # 1. Load Tokenizer
    tokenizer = get_tokenizer(config.MODEL_CHECKPOINT_SHORT)

    # 2. Load Raw Datasets
    raw_datasets = load_qna_datasets(
        config.SHORT_QNA_TRAIN_PATH,
        config.SHORT_QNA_VAL_PATH,
        qna_type="Short"
    )

    # 3. Preprocess and Tokenize Datasets
    if not raw_datasets["train"] or not raw_datasets["validation"]:
        print("Short Q&A training or validation dataset is empty. Exiting training.")
        return

    tokenized_datasets = tokenize_datasets(
        raw_datasets,
        tokenizer,
        config.INPUT_PREFIX_SHORT,
        config.OUTPUT_STRUCTURE_SHORT,
        config.MAX_INPUT_LENGTH_SHORT,
        config.MAX_TARGET_LENGTH_SHORT,
        qna_type="Short"
    )

    # 4. Load Base Model
    print(f"\nLoading base model '{config.MODEL_CHECKPOINT_SHORT}' for Short Q&A training...")
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_CHECKPOINT_SHORT)
    model.to(config.DEVICE)  # Move model to device
    print("Base model loaded and moved to device.")

    # 5. Training Arguments
    print(f"\nSetting up Training Arguments. Output dir: {config.SHORT_QNA_OUTPUT_DIR}")
    train_dataset_size = len(tokenized_datasets["train"])
    # num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    # if num_gpus == 0: num_gpus = 1 # Should not happen if DEVICE is CPU

    # effective_batch_size = config.BATCH_SIZE_SHORT * num_gpus
    # steps_per_epoch = (train_dataset_size // effective_batch_size) + (1 if train_dataset_size % effective_batch_size != 0 else 0)

    # Using logging_steps directly as save_steps if ratio calculation is too small
    # save_steps = int(steps_per_epoch * config.SAVE_STEPS_RATIO_SHORT)
    # if save_steps < 10: save_steps = config.LOGGING_STEPS_SHORT
    # save_strategy="steps" and save_steps is used if you want to save more frequently than epoch
    # The original notebook used save_strategy="epoch"

    args = Seq2SeqTrainingArguments(
        output_dir=config.SHORT_QNA_OUTPUT_DIR,
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        learning_rate=config.LEARNING_RATE_SHORT,
        per_device_train_batch_size=config.BATCH_SIZE_SHORT,
        per_device_eval_batch_size=config.BATCH_SIZE_SHORT * 2,  # Can be larger for eval
        weight_decay=config.WEIGHT_DECAY_SHORT,
        save_total_limit=3,  # Max checkpoints to keep
        num_train_epochs=config.NUM_TRAIN_EPOCHS_SHORT,
        predict_with_generate=True,
        fp16=config.FP16_TRAINING,
        logging_dir=os.path.join(config.SHORT_QNA_OUTPUT_DIR, "logs"),
        logging_strategy="steps",
        logging_steps=config.LOGGING_STEPS_SHORT,
        save_strategy="epoch",  # Save at the end of each epoch
        # save_steps=save_steps, # Only if save_strategy="steps"
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="eval_loss",  # Use eval_loss to determine the best model
        report_to=["tensorboard"],  # Or "all" to include wandb, etc.
        generation_max_length=config.MAX_TARGET_LENGTH_SHORT  # For evaluation generations
    )

    # 6. Data Collator
    print("Setting up Data Collator...")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    print("Data Collator setup complete.")

    # 7. Trainer
    print("\nSetting up Trainer for Short Q&A task...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,  # Pass tokenizer for saving purposes and metrics
        compute_metrics=lambda p: compute_qna_metrics(p, tokenizer)
    )
    print("Trainer setup complete.")

    # 8. Train
    print("\nStarting training for Short Q&A Generation...")
    try:
        train_result = trainer.train()
        print("\nTraining finished.")

        # Save the final model (which is the best if load_best_model_at_end=True)
        trainer.save_model()  # Saves to args.output_dir
        print(f"Final Short Q&A model (best model) saved to {config.SHORT_QNA_OUTPUT_DIR}")

        # Log and save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train_short", metrics)
        trainer.save_metrics("train_short", metrics)
        trainer.save_state()

        # Explicitly save to "best_model" subfolder for clarity (Trainer already does this if load_best_model_at_end=True)
        short_qna_best_model_path = os.path.join(config.SHORT_QNA_OUTPUT_DIR, "best_model")
        if not os.path.exists(short_qna_best_model_path):  # Should exist due to load_best_model_at_end
            os.makedirs(short_qna_best_model_path)
        # trainer.save_model(short_qna_best_model_path) # This might be redundant if output_dir is already the "best" due to load_best_model_at_end
        print(f"Best Short Q&A Model explicitly available at {short_qna_best_model_path}")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        if torch.cuda.is_available():
            print("Attempting to clear CUDA cache...")
            torch.cuda.empty_cache()
        # Potentially re-raise or handle as needed
        raise

    finally:
        if torch.cuda.is_available():
            print("Cleaning up model and trainer from GPU memory (if applicable)...")
            if 'model' in locals(): del model
            if 'trainer' in locals(): del trainer
            torch.cuda.empty_cache()

    print("\nTraining process complete for Short Q&A.")


if __name__ == "__main__":
    main()