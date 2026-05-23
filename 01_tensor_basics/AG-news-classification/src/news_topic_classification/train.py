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

logger = setup_logger(LOGS_DIR / Path(__file__).stem, mode="w")


def train_model(
    data: DataLoader,
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

        for x, y in data:
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
        logger.info(
            f"Epoch: {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}"
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

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    train_data, vocab, max_doc_length = prepare_data(train_df)
    model = NewsTopicClassifier(vocab_size=len(vocab)).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train_model(
        data=train_data,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=5,
    )
    save_model(MODEL_SAVE_PATH, model, optimizer, vocab, max_doc_length)


if __name__ == "__main__":
    main()
