import pandas as pd
import re
from pathlib import Path

from news_topic_classification.config import TRAIN_DATA_PATH, TEST_DATA_PATH, LOGS_DIR
from news_topic_classification.util.logger import setup_logger

logger = setup_logger(LOGS_DIR / Path(__file__).stem, mode="w")


def clean_text(text: str) -> str:
    """Cleans the input text by removing special characters, extra spaces, urls, html tags, and converting to lowercase."""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes column names by converting to lowercase and replacing spaces with underscores."""
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


def combine_title_description(df: pd.DataFrame) -> pd.DataFrame:
    """Combines the 'title' and 'description' columns into a single 'text' column."""
    df["text"] = df["title"] + " " + df["description"]
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the input DataFrame by normalizing columns, combining title and description, and cleaning the text."""
    df_copy = df.copy()

    df_copy = normalize_columns(df_copy)
    df_copy["class_index"] = df_copy["class_index"].astype(int) - 1
    df_copy = combine_title_description(df_copy)
    raw_text_sample = df_copy["text"].head().to_string()
    df_copy["text"] = df_copy["text"].apply(clean_text)
    report = [
        f"Sample raw text:\n{raw_text_sample}",
        f"Original data shape: {df_copy.shape}",
        "",
        f"Sample preprocessed text:\n{df_copy['text'].head()}",
        f"Preprocessed data shape: {df_copy.shape}",
    ]

    return df_copy, "\n".join(report)


def preprocess_report(report: list[str], name: str) -> None:
    logger.info(f"\nPreprocessing Report for {name}:\n" + report)
