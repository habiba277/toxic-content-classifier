import torch
import torch.nn as nn


class ToxicLSTM(nn.Module):

    def __init__(self, hidden_dim, output_dim,
                 embedding_matrix):

        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
        embedding_matrix,
        freeze=False
)

        self.lstm = nn.LSTM(
            embedding_matrix.shape[1],
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(
            hidden_dim * 2,
            output_dim
        )

    def forward(self, x):

        embedded = self.embedding(x)

        output, (hidden, cell) = self.lstm(embedded)

        hidden = torch.cat(
            (hidden[-2,:,:], hidden[-1,:,:]),
            dim=1
        )

        hidden = self.dropout(hidden)

        return self.fc(hidden)
    
print("CODE IS RUNNING")