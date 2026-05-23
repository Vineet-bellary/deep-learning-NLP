from pathlib import Path

# Main paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Data paths
TRAIN_DATA_PATH = DATA_DIR / "news_train_dataset.csv"
TEST_DATA_PATH = DATA_DIR / "news_test_dataset.csv"

# Model paths
MODEL_SAVE_PATH = MODEL_DIR / "news_topic_classifier.pt"

# Constant parameters
MAX_VOCAB_SIZE = 100000
BATCH_SIZE = 8
NUM_CLASSES = 4
DEFAULT_NUM_EPOCH = 100

# Class Map
CLASS_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}
