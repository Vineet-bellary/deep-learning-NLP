import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from news_topic_classification.config import (
    LOGS_DIR,
    TEST_DATA_PATH,
    BATCH_SIZE,
    MODEL_SAVE_PATH,
)
from news_topic_classification.util.load_data import (
    load_model_metadata,
    prepare_sample_text,
)
from news_topic_classification.util.preprocessing import preprocess_data
from news_topic_classification.util.logger import setup_logger

logger = setup_logger(LOGS_DIR / Path(__file__).stem, mode="w")


def evaluate(model, loader):
    total_correct = 0
    total_samples = 0
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for doc_tensor, label_tensor in loader:
            doc_tensor = doc_tensor.to(device)
            label_tensor = label_tensor.to(device)
            pred = model(doc_tensor)

            predicted_classes = torch.argmax(pred, dim=1)
            total_correct += (predicted_classes == label_tensor).sum().item()
            total_samples += label_tensor.size(0)

    accuracy = total_correct / total_samples
    logger.info(f"Evaluation Accuracy: {accuracy:.2f}")


def confusion_matrix(model, loader):
    # This function can be implemented to compute and log the confusion matrix
    pass


if __name__ == "__main__":
    model, vocab, max_doc_length = load_model_metadata(MODEL_SAVE_PATH)

    test_df = pd.read_csv(TEST_DATA_PATH)
    preprocessed_test_df, _unused_preprocess_report = preprocess_data(test_df)

    tensor_docs = [
        prepare_sample_text(text, vocab, max_doc_length)
        for text in preprocessed_test_df["text"].tolist()
    ]

    doc_tensor = torch.stack(tensor_docs)
    label_tensor = torch.tensor(preprocessed_test_df["class_index"])

    dataset = TensorDataset(doc_tensor, label_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    evaluate(model, dataloader)
