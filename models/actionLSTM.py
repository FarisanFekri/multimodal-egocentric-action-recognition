import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionLSTM(nn.Module):
    """Action LSTM model for action recognition.
    Based on an LSTM and a fully connected classifier.
    """
    def __init__(self, feature_dim, num_classes, num_clips=5) -> None:
        super(ActionLSTM, self).__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_clips = num_clips

        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=512, num_layers=1)

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out, hidden_state = self.lstm(x.permute([1, 0, 2]))
        out = self.classifier(hidden_state[-1])
        return out.squeeze(), {}