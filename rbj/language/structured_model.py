import torch
import torch.nn as nn


class SJGLanguageStructured(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=96,
        hidden_dim=192,

        num_intents=3,
        num_actions=3,
        num_directions=5,
        num_targets=2,
        num_subjects=2,

        max_distance=4,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # -------------------------------------------------
        # Shared representation
        # -------------------------------------------------

        self.dropout = nn.Dropout(
            0.20
        )

        # -------------------------------------------------
        # Semantic heads
        # -------------------------------------------------

        self.intent_head = nn.Linear(
            hidden_dim,
            num_intents
        )

        self.action_head = nn.Linear(
            hidden_dim,
            num_actions
        )

        self.direction_head = nn.Linear(
            hidden_dim,
            num_directions
        )

        self.target_head = nn.Linear(
            hidden_dim,
            num_targets
        )

        self.subject_head = nn.Linear(
            hidden_dim,
            num_subjects
        )

        self.distance_head = nn.Linear(
            hidden_dim,
            max_distance + 1
        )

    def forward(self, input_ids):

        embedded = self.embedding(
            input_ids
        )

        output, hidden = self.gru(
            embedded
        )

        # Último estado oculto
        representation = hidden[-1]

        representation = self.dropout(
            representation
        )

        return {
            "intent": self.intent_head(
                representation
            ),

            "action": self.action_head(
                representation
            ),

            "direction": self.direction_head(
                representation
            ),

            "target": self.target_head(
                representation
            ),

            "subject": self.subject_head(
                representation
            ),

            "distance": self.distance_head(
                representation
            ),
        }