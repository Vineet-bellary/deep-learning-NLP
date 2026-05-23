import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from news_topic_classification.config import (
    LOGS_DIR,
    MAX_VOCAB_SIZE,
    BATCH_SIZE,
    NUM_CLASSES,
)
from news_topic_classification.util.logger import setup_logger
from news_topic_classification.util.preprocessing import preprocess_data, clean_text
from news_topic_classification.model import NewsTopicClassifier

logger = setup_logger(LOGS_DIR / Path(__file__).stem, mode="w")

"""
    1. Build vocabulary
    2. Tokenize Docs
    3. Vectorize Docs
    4. Pad vectorized docs
    5. Create DataLoader object
"""


def build_vocab(df: pd.DataFrame, text_col: str, max_vocab_size: int) -> set[str]:
    """Builds a vocabulary set from the specified text column in the DataFrame."""
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }
    for text in df[text_col]:
        if len(vocab) >= max_vocab_size:
            break
        for word in str(text).split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def tokenize_docs(df: pd.DataFrame, text_col: str) -> list[list[str]]:
    """Tokenizes the specified text column in the DataFrame into a list of token lists."""
    return [str(text).split() for text in df[text_col]]


def vectorize_docs(tokenized_docs: list[list[str]], vocab: set[str]) -> list[list[int]]:
    """Converts tokenized documents into lists of indices based on the provided vocabulary."""
    vectorized_docs = [
        [vocab.get(token, vocab["<UNK>"]) for token in doc] for doc in tokenized_docs
    ]

    max_doc_length = max(len(doc) for doc in vectorized_docs)

    return vectorized_docs, max_doc_length


def pad_vectorized_docs(
    vectorized_docs: list[list[int]], max_doc_length: int
) -> list[list[int]]:
    """Pads vectorized documents to a specified maximum document length."""
    for doc in vectorized_docs:
        while len(doc) < max_doc_length:
            doc.append(0)
    return vectorized_docs


def create_dataloader(x: torch.Tensor, y: torch.Tensor, batch_size: int):
    """Creates a DataLoader object from vectorized documents and their corresponding labels."""
    dataset = torch.utils.data.TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def prepare_data(df: pd.DataFrame) -> tuple[DataLoader, set[str], int]:

    logger.info(f"Training data loaded with shape: {df.shape}")

    preprocessed_train_df, _unused_preprocess_report = preprocess_data(df)
    min_class_index = int(preprocessed_train_df["class_index"].min())
    max_class_index = int(preprocessed_train_df["class_index"].max())

    assert (
        min_class_index >= 0 and max_class_index < NUM_CLASSES
    ), f"label range invalid: {min_class_index}..{max_class_index}"

    logger.info(
        f"Class index range: {min_class_index} to {max_class_index} (total classes: {NUM_CLASSES})"
    )

    vocab = build_vocab(
        preprocessed_train_df, text_col="text", max_vocab_size=MAX_VOCAB_SIZE
    )
    logger.info(f"Vocabulary size: {len(vocab)}")

    tokenized_docs = tokenize_docs(preprocessed_train_df, text_col="text")
    vectorized_docs, max_doc_length = vectorize_docs(tokenized_docs, vocab)

    padded_docs = pad_vectorized_docs(vectorized_docs, max_doc_length=max_doc_length)

    label_tensor = torch.tensor(preprocessed_train_df["class_index"])
    vectorized_tensor = torch.tensor(padded_docs)

    dataloader = create_dataloader(
        x=vectorized_tensor, y=label_tensor, batch_size=BATCH_SIZE
    )

    return dataloader, vocab, max_doc_length


def prepare_sample_text(
    sample_text: str, vocab: set[str], max_doc_length: int
) -> torch.Tensor:
    """Prepares a sample text for inference by tokenizing, vectorizing, and padding it."""
    cleaned_text = clean_text(sample_text)
    tokenized_text = cleaned_text.split()
    # vectorized_text = [vocab.get(token, vocab["<UNK>"]) for token in tokenized_text]
    vectorized_text = vectorize_docs([tokenized_text], vocab)[0][0]
    padded_vectorized_text = pad_vectorized_docs([vectorized_text], max_doc_length)[0]

    return torch.tensor(padded_vectorized_text)


def load_model_metadata(path: Path) -> NewsTopicClassifier:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state = torch.load(path, map_location=device)

    vocab = model_state["vocab"]
    max_doc_length = model_state["max_doc_length"]
    model_state_dict = model_state["model_state_dict"]

    model = NewsTopicClassifier(vocab_size=len(vocab), num_classes=NUM_CLASSES)
    model.load_state_dict(model_state_dict)
    model.to(device)

    return model, vocab, max_doc_length
