import os
import torch

# --- General Configuration ---
BASE_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_PROJECT_DIR, "data")
BASE_OUTPUT_DIR = os.path.join(BASE_PROJECT_DIR, "model_outputs")

# Ensure base output directory exists
if not os.path.exists(BASE_OUTPUT_DIR):
    os.makedirs(BASE_OUTPUT_DIR)
    print(f"Created base output directory: {BASE_OUTPUT_DIR}")

# --- Short Q&A Configuration ---
MODEL_CHECKPOINT_SHORT = "t5-small"

SHORT_QNA_TRAIN_PATH = os.path.join(DATA_DIR, "short_train.csv")
SHORT_QNA_VAL_PATH = os.path.join(DATA_DIR, "short_val.csv")
SHORT_QNA_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "short_qna_finetuned")

# Short Q&A Preprocessing Parameters
MAX_INPUT_LENGTH_SHORT = 750
MAX_TARGET_LENGTH_SHORT = 128
INPUT_PREFIX_SHORT = "generate question and answer: context: "
OUTPUT_STRUCTURE_SHORT = "question: {} answer: {}"

# Short Q&A Training Parameters
BATCH_SIZE_SHORT = 12
LEARNING_RATE_SHORT = 5e-5
NUM_TRAIN_EPOCHS_SHORT = 6
WEIGHT_DECAY_SHORT = 0.01
LOGGING_STEPS_SHORT = 100
SAVE_STEPS_RATIO_SHORT = 0.2 # This ratio is used to calculate save_steps based on epoch size

# Short Q&A Inference Parameters (used by inference_utils.py and app.py)
SHORT_QNA_MODEL_LOAD_PATH = os.path.join(SHORT_QNA_OUTPUT_DIR, "best_model")
INF_MAX_OUTPUT_LENGTH_SHORT = 128
INF_NUM_BEAMS_SHORT_DEFAULT = 4 # Default for simple inference script
INF_EARLY_STOPPING_SHORT_DEFAULT = True
INF_NO_REPEAT_NGRAM_SIZE_SHORT_DEFAULT = 2

# For Flask App - more diverse generation
FLASK_INF_NUM_BEAMS_SHORT = 1
FLASK_INF_EARLY_STOPPING_SHORT = False # Less relevant with num_beams=1 and do_sample=True
FLASK_INF_NO_REPEAT_NGRAM_SIZE_SHORT = 3
FLASK_NUM_RETURN_SEQUENCES_SHORT = 3
FLASK_TEMPERATURE_SHORT = 0.9
FLASK_TOP_P_SHORT = 0.9
FLASK_TOP_K_SHORT = 0


# --- Long Q&A Configuration ---
MODEL_CHECKPOINT_LONG = "t5-small"

LONG_QNA_TRAIN_PATH = os.path.join(DATA_DIR, "long_train.csv")
LONG_QNA_VAL_PATH = os.path.join(DATA_DIR, "long_val.csv")
LONG_QNA_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "long_qna_finetuned")

# Long Q&A Preprocessing Parameters
MAX_INPUT_LENGTH_LONG = 750
MAX_TARGET_LENGTH_LONG = 256
INPUT_PREFIX_LONG = "generate question and answer: context: "
OUTPUT_STRUCTURE_LONG = "question: {} answer: {}"

# Long Q&A Training Parameters
BATCH_SIZE_LONG = 8
LEARNING_RATE_LONG = 5e-5
NUM_TRAIN_EPOCHS_LONG = 6
WEIGHT_DECAY_LONG = 0.01
LOGGING_STEPS_LONG = 100
SAVE_STEPS_RATIO_LONG = 0.2

# Long Q&A Inference Parameters (used by inference_utils.py and app.py)
LONG_QNA_MODEL_LOAD_PATH = os.path.join(LONG_QNA_OUTPUT_DIR, "best_model")
INF_MAX_OUTPUT_LENGTH_LONG = 256
INF_NUM_BEAMS_LONG_DEFAULT = 4 # Default for simple inference script
INF_EARLY_STOPPING_LONG_DEFAULT = True
INF_NO_REPEAT_NGRAM_SIZE_LONG_DEFAULT = 2

# For Flask App - more diverse generation
FLASK_INF_NUM_BEAMS_LONG = 1
FLASK_INF_EARLY_STOPPING_LONG = False
FLASK_INF_NO_REPEAT_NGRAM_SIZE_LONG = 3
FLASK_NUM_RETURN_SEQUENCES_LONG = 3
FLASK_TEMPERATURE_LONG = 0.9
FLASK_TOP_P_LONG = 0.9
FLASK_TOP_K_LONG = 0


# --- General Training & Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FP16_TRAINING = torch.cuda.is_available()

# --- Flask App Configuration ---
FLASK_WORD_LIMIT = 300
# For ngrok, consider using environment variables for sensitive data
# NGROK_AUTHTOKEN = "YOUR_NGROK_AUTHTOKEN" # Example: "2xkc0M26NfumNL9ASRxPB1ji1Ak_4J9UXseie1BgEDBUCMUc1"


def print_short_qna_config():
    print(f"Configuration for SHORT Q&A:")
    print(f"  Model Checkpoint: {MODEL_CHECKPOINT_SHORT}")
    print(f"  Training CSV: {SHORT_QNA_TRAIN_PATH}")
    print(f"  Validation CSV: {SHORT_QNA_VAL_PATH}")
    print(f"  Output Directory (for checkpoints & logs): {SHORT_QNA_OUTPUT_DIR}")
    print(f"  Max Input Length: {MAX_INPUT_LENGTH_SHORT}")
    print(f"  Max Target Length: {MAX_TARGET_LENGTH_SHORT}")
    print(f"  Batch Size: {BATCH_SIZE_SHORT}")
    print(f"  Epochs: {NUM_TRAIN_EPOCHS_SHORT}")
    print(f"  Device: {DEVICE}")

def print_long_qna_config():
    print(f"Configuration for LONG Q&A:")
    print(f"  Model Checkpoint: {MODEL_CHECKPOINT_LONG}")
    print(f"  Training CSV: {LONG_QNA_TRAIN_PATH}")
    print(f"  Validation CSV: {LONG_QNA_VAL_PATH}")
    print(f"  Output Directory (for checkpoints & logs): {LONG_QNA_OUTPUT_DIR}")
    print(f"  Max Input Length: {MAX_INPUT_LENGTH_LONG}")
    print(f"  Max Target Length: {MAX_TARGET_LENGTH_LONG}")
    print(f"  Batch Size: {BATCH_SIZE_LONG}")
    print(f"  Epochs: {NUM_TRAIN_EPOCHS_LONG}")
    print(f"  Device: {DEVICE}")