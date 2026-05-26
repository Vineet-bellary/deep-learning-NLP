import pandas as pd
from sklearn.model_selection import train_test_split

from news_topic_classification.util.preprocessing import normalize_columns


def split_train_val(
    df: pd.DataFrame, label_col: str, val_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the DataFrame into training and validation sets while maintaining class distribution."""
    train_df, val_df = train_test_split(
        df, test_size=val_size, stratify=df[label_col], random_state=random_state
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
