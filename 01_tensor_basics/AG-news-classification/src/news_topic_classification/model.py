import torch
import torch.nn as nn

from news_topic_classification.config import NUM_CLASSES


class NewsTopicClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int = NUM_CLASSES):
        super(NewsTopicClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=5)
        self.layer1 = nn.Linear(in_features=5, out_features=8)
        self.layer2 = nn.Linear(in_features=8, out_features=6)
        self.layer3 = nn.Linear(in_features=6, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)

        return x
