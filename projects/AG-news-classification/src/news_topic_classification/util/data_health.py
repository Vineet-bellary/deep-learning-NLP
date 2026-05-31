import pandas as pd
from pathlib import Path

from news_topic_classification.config import TRAIN_DATA_PATH, TEST_DATA_PATH, LOGS_DIR
from news_topic_classification.util.logger import setup_logger

logger = setup_logger(LOGS_DIR / Path(__file__).stem, mode="w")


def obs_data(df: pd.DataFrame) -> None:

    logger.info(f"Shape: {df.shape}\nData:\n{df.head()}\n")


def get_columns(df: pd.DataFrame) -> None:
    return df.columns.to_list()


def check_missing_values(df: pd.DataFrame) -> None:
    logger.info(f"Missing values in data:\n{df.isnull().sum()}")


def check_duplicates(df: pd.DataFrame) -> None:
    logger.info(f"Duplicate rows in data: {df.duplicated().sum()}")


def check_class_distribution(df: pd.DataFrame, label_col: list[str]) -> None:
    logger.info(f"Class distribution:\n{df[label_col].value_counts()}")


def get_health_report(df: pd.DataFrame, df_name: str) -> None:
    """Generates a cleanly formatted, cohesive string report of the dataset."""

    shape_info = f"{df.shape[0]} rows, {df.shape[1]} columns"
    missing_vals = df.isnull().sum().to_string()
    duplicates = df.duplicated().sum()

    label_col = df.columns[0]
    class_dist = df[label_col].value_counts().to_string()

    df_head = df.head().to_string()

    report = [
        f"\n{'='*30} {df_name.upper()} DATA HEALTH REPORT {'='*30}",
        f"Dimensions:     {shape_info}",
        f"Columns:        {df.columns.tolist()}",
        f"Duplicate Rows: {duplicates}",
        f"\n[ MISSING VALUES ]\n{missing_vals}",
        f"\n[ CLASS DISTRIBUTION ({label_col}) ]\n{class_dist}",
        f"\n[ DATA PREVIEW (FIRST 5 ROWS) ]\n{df_head}",
        f"{'='*80}\n",
    ]

    return "\n".join(report)


def main():
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)

    logger.info(get_health_report(train_data, "Train"))
    logger.info(get_health_report(test_data, "Test"))


if __name__ == "__main__":
    main()
