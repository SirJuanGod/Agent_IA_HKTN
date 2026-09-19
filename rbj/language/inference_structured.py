import torch
from pathlib import Path
from sys import path
path.append(str(Path(__file__).parent / ".." / ".."))

from rbj.language.vocabulary import Vocabulary
from rbj.language.tokenizer import Tokenizer
from rbj.language.structured_model import (
    SJGLanguageStructured,
)


class SJGLanguageStructuredInference:

    def __init__(
        self,
        checkpoint_path,
        device=None,
    ):

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = torch.device(
                device
            )

        # -------------------------------------------------
        # Checkpoint
        # -------------------------------------------------

        self.checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        # -------------------------------------------------
        # Vocabulary
        # -------------------------------------------------

        self.vocab = Vocabulary()

        self.vocab.word_to_id = (
            self.checkpoint[
                "vocab_state"
            ]
        )

        self.vocab.id_to_word = {
            int(index): word
            for word, index
            in self.vocab.word_to_id.items()
        }

        self.tokenizer = Tokenizer(
            self.vocab
        )

        # -------------------------------------------------
        # Label maps
        # -------------------------------------------------

        self.label_maps = (
            self.checkpoint[
                "label_maps"
            ]
        )

        self.reverse_maps = {}

        for field, mapping in (
            self.label_maps.items()
        ):

            self.reverse_maps[field] = {
                index: label
                for label, index
                in mapping.items()
            }

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        self.model = SJGLanguageStructured(

            vocab_size=len(
                self.vocab
            ),

            embedding_dim=self.checkpoint[
                "embedding_dim"
            ],

            hidden_dim=self.checkpoint[
                "hidden_dim"
            ],

            num_intents=len(
                self.label_maps["intent"]
            ),

            num_actions=len(
                self.label_maps["action"]
            ),

            num_directions=len(
                self.label_maps["direction"]
            ),

            num_targets=len(
                self.label_maps["target"]
            ),

            num_subjects=len(
                self.label_maps["subject"]
            ),

            max_distance=self.checkpoint[
                "max_distance"
            ],

        ).to(self.device)

        self.model.load_state_dict(
            self.checkpoint[
                "model_state_dict"
            ]
        )

        self.model.eval()

        self.max_length = (
            self.checkpoint[
                "max_length"
            ]
        )

    # =====================================================
    # TOKENIZATION
    # =====================================================

    def _encode(self, text):

        token_ids = self.tokenizer.encode(
            text
        )

        if len(token_ids) < self.max_length:

            token_ids += [
                self.vocab.pad_id
            ] * (
                self.max_length
                - len(token_ids)
            )

        else:

            token_ids = token_ids[
                :self.max_length
            ]

        return torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )

    # =====================================================
    # PREDICTION
    # =====================================================

    @torch.no_grad()
    def predict(self, text):

        inputs = self._encode(
            text
        )

        outputs = self.model(
            inputs
        )

        result = {
            "text": text,
        }

        # -------------------------------------------------
        # Cada cabeza
        # -------------------------------------------------

        for field in [
            "intent",
            "action",
            "direction",
            "target",
            "subject",
        ]:

            probabilities = torch.softmax(
                outputs[field],
                dim=1,
            )

            confidence, prediction = (
                probabilities.max(
                    dim=1
                )
            )

            prediction_id = (
                prediction.item()
            )

            result[field] = (
                self.reverse_maps[field][
                    prediction_id
                ]
            )

            result[
                f"{field}_confidence"
            ] = confidence.item()

        # -------------------------------------------------
        # Distance
        # -------------------------------------------------

        distance_probabilities = (
            torch.softmax(
                outputs["distance"],
                dim=1,
            )
        )

        distance_confidence, distance = (
            distance_probabilities.max(
                dim=1
            )
        )

        result["distance"] = (
            distance.item()
        )

        result[
            "distance_confidence"
        ] = distance_confidence.item()

        return result

    # =====================================================
    # TOP-K
    # =====================================================

    @torch.no_grad()
    def top_k(
        self,
        text,
        field,
        k=3,
    ):

        inputs = self._encode(
            text
        )

        outputs = self.model(
            inputs
        )

        probabilities = torch.softmax(
            outputs[field],
            dim=1,
        )

        values, indices = torch.topk(
            probabilities,
            k=min(
                k,
                probabilities.size(1)
            ),
            dim=1,
        )

        results = []

        for value, index in zip(
            values[0],
            indices[0]
        ):

            index = index.item()

            if field == "distance":

                label = index

            else:

                label = self.reverse_maps[
                    field
                ][index]

            results.append({
                "value": label,
                "confidence": value.item(),
            })

        return results