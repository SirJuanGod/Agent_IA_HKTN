import torch

from .vocabulary import Vocabulary
from .tokenizer import Tokenizer
from .dataset import IntentVocabulary
from .model import SJGLanguage


class SJGLanguageInference:
    """
    Interfaz de inferencia para SJG-Agent.

    Carga un modelo entrenado y permite convertir
    texto natural en una intención.
    """

    def __init__(
        self,
        checkpoint_path,
        device=None,
    ):

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        # --------------------------------------------------
        # Cargar checkpoint
        # --------------------------------------------------

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        # --------------------------------------------------
        # Vocabulary
        # --------------------------------------------------

        self.vocab = Vocabulary()

        self.vocab.word_to_id = (
            checkpoint["vocab_state"]
        )

        self.vocab.id_to_word = {
            idx: word
            for word, idx
            in self.vocab.word_to_id.items()
        }

        # --------------------------------------------------
        # Tokenizer
        # --------------------------------------------------

        self.tokenizer = Tokenizer(
            self.vocab
        )

        # --------------------------------------------------
        # Intent vocabulary
        # --------------------------------------------------

        self.intent_vocab = IntentVocabulary()

        self.intent_vocab.word_to_id = (
            checkpoint["intent_state"]
        )

        self.intent_vocab.id_to_word = {
            idx: intent
            for intent, idx
            in self.intent_vocab.word_to_id.items()
        }

        # --------------------------------------------------
        # Configuración del modelo
        # --------------------------------------------------

        self.embedding_dim = checkpoint[
            "embedding_dim"
        ]

        self.hidden_dim = checkpoint[
            "hidden_dim"
        ]

        self.max_length = checkpoint[
            "max_length"
        ]

        # --------------------------------------------------
        # Modelo
        # --------------------------------------------------

        self.model = SJGLanguage(
            vocab_size=len(self.vocab),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=len(self.intent_vocab),
            padding_idx=self.vocab.word_to_id[
                "<PAD>"
            ],
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)

        self.model.eval()

    # ======================================================
    # Preparar texto
    # ======================================================

    def _prepare_text(self, text):

        encoded = self.tokenizer.encode(
            text
        )

        # Limitar longitud
        encoded = encoded[:self.max_length]

        # Padding
        pad_id = self.vocab.word_to_id[
            "<PAD>"
        ]

        padding_length = (
            self.max_length - len(encoded)
        )

        encoded += (
            [pad_id] * padding_length
        )

        return encoded

    # ======================================================
    # Predecir
    # ======================================================

    def predict(self, text):

        input_ids = self._prepare_text(
            text
        )

        tensor = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():

            logits = self.model(
                tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1,
            )

        prediction_id = prediction.item()

        confidence_value = confidence.item()

        intent = self.intent_vocab.decode(
            prediction_id
        )

        return {
            "text": text,
            "intent": intent,
            "intent_id": prediction_id,
            "confidence": confidence_value,
        }

    # ======================================================
    # Top-K
    # ======================================================

    def predict_top_k(
        self,
        text,
        k=3,
    ):

        input_ids = self._prepare_text(
            text
        )

        tensor = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():

            logits = self.model(
                tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            values, indices = torch.topk(
                probabilities,
                k=min(k, len(self.intent_vocab)),
                dim=1,
            )

        results = []

        for probability, index in zip(
            values[0],
            indices[0],
        ):

            intent_id = index.item()

            results.append(
                {
                    "intent": self.intent_vocab.decode(
                        intent_id
                    ),
                    "intent_id": intent_id,
                    "confidence": probability.item(),
                }
            )

        return results