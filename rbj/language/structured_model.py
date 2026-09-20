import torch
import torch.nn as nn


class SJGLanguageStructured(nn.Module):
    """
    Modelo de lenguaje estructurado de SJG-Agent.

    Mejoras respecto a la versión anterior:
      - GRU bicapa con dropout entre capas.
      - LayerNorm en la representación compartida.
      - Cada head semántico usa Sequential(Linear → GELU → Linear)
        en lugar de un Linear simple, lo que proporciona la función
        de activación más agresiva requerida para mejorar la estimación
        de la intención del comando.

    Arquitectura:

        Token IDs
            ↓
        Embedding
            ↓
        GRU (2 capas, dropout=0.2)
            ↓
        Hidden State [-1]
            ↓
        LayerNorm
            ↓
        Dropout
            ↓
        ┌──────────────────────────────────────┐
        │  intent_head  → Linear→GELU→Linear   │
        │  action_head  → Linear→GELU→Linear   │
        │  direction_head→ Linear→GELU→Linear  │
        │  target_head  → Linear→GELU→Linear   │
        │  subject_head → Linear→GELU→Linear   │
        │  distance_head→ Linear→GELU→Linear   │
        └──────────────────────────────────────┘
    """

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

        gru_layers=2,
        dropout=0.20,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # --------------------------------------------------
        # GRU multicapa con dropout entre capas
        # --------------------------------------------------

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # --------------------------------------------------
        # Normalización y regularización en la representación
        # compartida antes de los heads
        # --------------------------------------------------

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # --------------------------------------------------
        # Función auxiliar para construir un head semántico
        # con activación GELU (más agresiva que ReLU para
        # capturar intenciones complejas)
        # --------------------------------------------------

        def _make_head(in_dim: int, out_dim: int) -> nn.Sequential:
            mid_dim = max(in_dim // 2, out_dim)
            return nn.Sequential(
                nn.Linear(in_dim, mid_dim),
                nn.GELU(),
                nn.Linear(mid_dim, out_dim),
            )

        # --------------------------------------------------
        # Heads semánticos
        # --------------------------------------------------

        self.intent_head = _make_head(hidden_dim, num_intents)
        self.action_head = _make_head(hidden_dim, num_actions)
        self.direction_head = _make_head(hidden_dim, num_directions)
        self.target_head = _make_head(hidden_dim, num_targets)
        self.subject_head = _make_head(hidden_dim, num_subjects)
        self.distance_head = _make_head(hidden_dim, max_distance + 1)

    def forward(self, input_ids):
        """
        Args:
            input_ids: Tensor de forma [batch_size, seq_len]

        Returns:
            dict con logits para cada campo semántico.
        """

        embedded = self.embedding(input_ids)

        output, hidden = self.gru(embedded)

        # Usar el último estado oculto de la última capa
        representation = hidden[-1]  # [B, H]

        representation = self.layer_norm(representation)

        representation = self.dropout(representation)

        return {
            "intent":    self.intent_head(representation),
            "action":    self.action_head(representation),
            "direction": self.direction_head(representation),
            "target":    self.target_head(representation),
            "subject":   self.subject_head(representation),
            "distance":  self.distance_head(representation),
        }