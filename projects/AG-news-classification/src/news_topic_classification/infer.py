import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from news_topic_classification.config import (
    MODEL_SAVE_PATH,
    TEST_DATA_PATH,
    LOGS_DIR,
    CLASS_MAP,
)
from news_topic_classification.util.load_data import (
    prepare_sample_text,
    load_model_metadata,
)
from news_topic_classification.util.preprocessing import (
    preprocess_data,
    preprocess_report,
)
from news_topic_classification.util.logger import setup_logger

logger = setup_logger(LOGS_DIR / Path(__file__).stem, mode="w")


def infer(
    model, vocab: set[str], max_doc_length: int, sample_text: list[str]
) -> dict[str, str]:
    model.eval()
    device = next(model.parameters()).device
    results = {}
    for sample in sample_text:
        with torch.no_grad():
            sample_tensor = (
                prepare_sample_text(sample, vocab, max_doc_length)
                .unsqueeze(dim=0)
                .to(device)
            )
            output = model(sample_tensor)
            predicted_class_id = torch.argmax(output, dim=1).item()
            predicted_class = CLASS_MAP[predicted_class_id]
            results[sample] = predicted_class
    return results


if __name__ == "__main__":
    model, vocab, max_doc_length = load_model_metadata(MODEL_SAVE_PATH)

    test_df = pd.read_csv(TEST_DATA_PATH)
    cleaned_test_df, report = preprocess_data(test_df)
    preprocess_report(report, "Test Data")

    test_samples = cleaned_test_df["text"].head(10).tolist()

    predicted_classes = infer(model, vocab, max_doc_length, test_samples)
    logger.info(f"Inference by {model._get_name()}:\n")
    res = "\n".join(
        [
            f"{sample[:45]}... : {predicted_class}\n"
            for sample, predicted_class in predicted_classes.items()
        ]
    )
    logger.info("\n" + res)
