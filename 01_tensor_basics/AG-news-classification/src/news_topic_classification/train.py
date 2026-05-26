import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from news_topic_classification.config import (
    DEFAULT_NUM_EPOCH,
    TRAIN_DATA_PATH,
    MODEL_SAVE_PATH,
    LOGS_DIR,
)
from news_topic_classification.util.load_data import prepare_data
from news_topic_classification.model import NewsTopicClassifier
from news_topic_classification.util.logger import setup_logger
from news_topic_classification.util.train_val_split import split_train_val
from news_topic_classification.util.preprocessing import preprocess_data

logger = setup_logger(LOGS_DIR / Path(__file__).stem, mode="w")


def val_model(data: DataLoader, model, loss_fn, device: torch.device):
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_samples = 0

    with torch.no_grad():

        for x, y in data:

            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            loss = loss_fn(pred, y)

            predicted_classes = torch.argmax(pred, dim=1)

            val_correct += (predicted_classes == y).sum().item()

            val_samples += y.size(0)

            val_loss += loss.item() * y.size(0)

    val_loss /= val_samples

    val_accuracy = val_correct / val_samples

    return val_loss, val_accuracy


def train_model(
    train_data: DataLoader,
    val_data: DataLoader,
    model,
    optimizer,
    loss_fn,
    device: torch.device,
    epochs: int = DEFAULT_NUM_EPOCH,
):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for x, y in train_data:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicted_classes = torch.argmax(pred, dim=1)
            epoch_correct += (predicted_classes == y).sum().item()
            epoch_samples += y.size(0)
            epoch_loss += loss.item() * y.size(0)

        epoch_loss /= epoch_samples
        epoch_accuracy = epoch_correct / epoch_samples

        val_loss, val_accuracy = val_model(val_data, model, loss_fn, device)

        logger.info(
            f"Epoch: {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}"
        )


def save_model(path: Path, model, optimizer, vocab, max_doc_length):
    model_save = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab": vocab,
        "max_doc_length": max_doc_length,
    }
    torch.save(model_save, path)
    logger.info(f"Model saved to {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f"Using device: {device}")

    data = pd.read_csv(TRAIN_DATA_PATH)

    train_df, val_df = split_train_val(data, label_col="class_index")

    train_data, vocab, max_doc_length = prepare_data(train_df)
    val_data, _unused_vocab, _unused_max_doc_length = prepare_data(val_df, vocab=vocab)

    model = NewsTopicClassifier(vocab_size=len(vocab)).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train_model(
        train_data=train_data,
        val_data=val_data,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=10,
    )
    save_model(MODEL_SAVE_PATH, model, optimizer, vocab, max_doc_length)


if __name__ == "__main__":
    main()
