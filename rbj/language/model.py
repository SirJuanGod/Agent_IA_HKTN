import torch
import torch.nn as nn


class SJGLanguage(nn.Module):
    """
    Modelo de lenguaje básico de SJG-Agent.

    Arquitectura:

        Token IDs
            ↓
        Embedding
            ↓
        GRU
            ↓
        Hidden State
            ↓
        Linear
            ↓
        Intent logits
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        num_classes=6,
        padding_idx=0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.padding_idx = padding_idx

        # --------------------------------------------------
        # Embedding
        # --------------------------------------------------

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        # --------------------------------------------------
        # GRU
        # --------------------------------------------------

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # --------------------------------------------------
        # Clasificador
        # --------------------------------------------------

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes,
        )

    def forward(self, input_ids):
        """
        input_ids:

            [batch_size, sequence_length]

        retorna:

            [batch_size, num_classes]
        """

        # [B, T] → [B, T, E]
        embedded = self.embedding(input_ids)

        # [B, T, E] → [B, T, H]
        output, hidden = self.gru(embedded)

        # Ignorar tokens de padding tomando el promedio de los outputs reales
        mask = (input_ids != self.padding_idx).float().unsqueeze(-1) # [B, T, 1]
        masked_output = output * mask
        summed_output = torch.sum(masked_output, dim=1)
        lengths = torch.clamp(torch.sum(mask, dim=1), min=1.0)
        mean_hidden = summed_output / lengths

        # [B, H] → [B, classes]
        logits = self.classifier(mean_hidden)

        return logits

    def predict(self, input_ids):
        """
        Realiza una predicción.

        Retorna los IDs de las clases.
        """

        self.eval()

        with torch.no_grad():

            logits = self.forward(input_ids)

            predictions = torch.argmax(
                logits,
                dim=1,
            )

        return predictions