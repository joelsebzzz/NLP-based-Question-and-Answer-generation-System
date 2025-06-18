import nltk
import numpy as np
import evaluate  # evaluate library for ROUGE

# Global variable to store loaded rouge_metric to avoid reloading
_rouge_metric = None


def get_rouge_metric():
    global _rouge_metric
    if _rouge_metric is None:
        print("\nSetting up evaluation metrics (ROUGE)...")
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            print("Downloading NLTK punkt tokenizer for metrics...")
            nltk.download('punkt', quiet=True)
        try:  # Also ensure punkt_tab for sentence tokenization in metrics
            nltk.data.find('tokenizers/punkt_tab')
        except nltk.downloader.DownloadError:
            print("Downloading NLTK punkt_tab tokenizer for metrics...")
            nltk.download('punkt_tab', quiet=True)

        _rouge_metric = evaluate.load("rouge")
        print("Metrics setup complete.")
    return _rouge_metric


def compute_qna_metrics(eval_pred, tokenizer):
    """Computes ROUGE scores and generation length for Q&A tasks."""
    rouge_metric = get_rouge_metric()
    predictions, labels = eval_pred

    # Replace -100 in predictions and labels with pad_token_id for decoding
    predictions = np.where(predictions == -100, tokenizer.pad_token_id, predictions)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # NLTK sentence tokenization for ROUGE (as in original notebook)
    # This expects punkt_tab to be available.
    decoded_preds_nltk = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels_nltk = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    rouge_result = rouge_metric.compute(predictions=decoded_preds_nltk, references=decoded_labels_nltk,
                                        use_stemmer=True)
    rouge_result = {key: value * 100 for key, value in
                    rouge_result.items()}  # Score is usually a float 0-1, scale to 0-100

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    metrics = {**rouge_result, "gen_len": np.mean(prediction_lens)}

    return {k: round(v, 4) for k, v in metrics.items()}