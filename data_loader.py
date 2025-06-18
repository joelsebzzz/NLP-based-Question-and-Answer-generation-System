import os
import pandas as pd
from datasets import Dataset, DatasetDict


def load_qna_datasets(train_path, val_path, qna_type="Short"):
    """Loads Q&A datasets from CSV files."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{qna_type} Q&A Training file not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"{qna_type} Q&A Validation file not found: {val_path}")

    print(f"Loading custom {qna_type} Q&A datasets from CSV using pandas...")
    try:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)

        raw_datasets = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
        print(f"Successfully loaded {qna_type} Q&A datasets using pandas.")

        print(f"\n{qna_type} Q&A Dataset structure:")
        print(raw_datasets)
        print(f"\nSample {qna_type} Q&A training example:")
        if len(raw_datasets["train"]) > 0:
            print(raw_datasets["train"][0])
        else:
            print(f"Warning: {qna_type} Q&A training dataset is empty.")

        # Column validation
        required_columns = ['context', 'question', 'answer']
        for split in raw_datasets.keys():
            if len(raw_datasets[split]) > 0:
                for col in required_columns:
                    if col not in raw_datasets[split].column_names:
                        # Attempt to strip column names if there's a mismatch due to whitespace
                        stripped_column_names = [c.strip() for c in raw_datasets[split].column_names]
                        if col not in stripped_column_names:
                            raise ValueError(
                                f"Missing required column '{col}' in {qna_type} Q&A '{split}' split. "
                                f"Available columns: {raw_datasets[split].column_names}"
                            )
            else:
                print(f"Warning: {qna_type} Q&A '{split}' split is empty. Skipping column check.")

        return raw_datasets

    except Exception as e:
        print(f"Error loading {qna_type} Q&A CSVs with pandas: {e}")
        print("Please ensure your CSV files are correctly formatted and paths are correct.")
        raise