import random

import torch
from torch.utils.data import Dataset


class IntentVocabulary:
    """
    Mapeo entre nombres de intención e IDs.
    """

    INTENTS = [
        "GO_TO_GOAL",
        "MOVE_UP",
        "MOVE_DOWN",
        "MOVE_LEFT",
        "MOVE_RIGHT",
        "STOP",
    ]

    def __init__(self):
        self.word_to_id = {
            intent: i
            for i, intent in enumerate(self.INTENTS)
        }

        self.id_to_word = {
            i: intent
            for intent, i in self.word_to_id.items()
        }

    def encode(self, intent):
        return self.word_to_id[intent]

    def decode(self, idx):
        return self.id_to_word[idx]

    def __len__(self):
        return len(self.INTENTS)


class LanguageDataset(Dataset):
    """
    Dataset lingüístico para clasificación de intenciones.
    """

    def __init__(
        self,
        sentences,
        labels,
        tokenizer,
        max_length=None,
    ):

        if len(sentences) != len(labels):
            raise ValueError(
                "sentences y labels deben tener la misma longitud."
            )

        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer

        self.encoded = [
            tokenizer.encode(sentence)
            for sentence in sentences
        ]

        if max_length is None:
            max_length = max(
                len(sequence)
                for sequence in self.encoded
            )

        self.max_length = max_length

    def _pad(self, sequence):

        sequence = sequence[:self.max_length]

        pad_id = self.tokenizer.vocabulary.word_to_id[
            "<PAD>"
        ]

        padding_length = (
            self.max_length - len(sequence)
        )

        sequence += [
            pad_id
        ] * padding_length

        return sequence

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):

        sequence = self._pad(
            self.encoded[index]
        )

        return {
            "input_ids": torch.tensor(
                sequence,
                dtype=torch.long,
            ),
            "label": torch.tensor(
                self.labels[index],
                dtype=torch.long,
            ),
        }


def generate_language_data():

    data = {

        # ==================================================
        # GO TO GOAL
        # ==================================================

        "GO_TO_GOAL": [

            "ve hacia la meta",
            "ve a la meta",
            "ve hasta la meta",
            "dirígete a la meta",
            "dirigete a la meta",
            "dirígete hacia la meta",
            "dirigete hacia la meta",
            "camina hacia la meta",
            "camina hasta la meta",
            "avanza hacia la meta",
            "avanza hasta la meta",
            "llega a la meta",
            "llega hasta la meta",
            "alcanza la meta",
            "busca la meta",
            "encuentra la meta",
            "ve hacia el objetivo",
            "ve al objetivo",
            "ve hasta el objetivo",
            "dirígete al objetivo",
            "dirigete al objetivo",
            "dirígete hacia el objetivo",
            "dirigete hacia el objetivo",
            "camina hacia el objetivo",
            "camina hasta el objetivo",
            "avanza hacia el objetivo",
            "avanza hasta el objetivo",
            "llega al objetivo",
            "llega hasta el objetivo",
            "alcanza el objetivo",
            "busca el objetivo",
            "encuentra el objetivo",
            "quiero que vayas a la meta",
            "quiero que vayas al objetivo",
            "ve directamente a la meta",
            "ve directamente al objetivo",
            "dirige el robot a la meta",
            "dirige el robot al objetivo",
            "lleva el robot a la meta",
            "lleva el robot al objetivo",
            "haz que el robot llegue a la meta",
            "haz que el robot llegue al objetivo",
            "mueve el robot hacia la meta",
            "mueve el robot hacia el objetivo",
            "avanza hasta encontrar la meta",
            "avanza hasta encontrar el objetivo",
        ],

        # ==================================================
        # MOVE UP
        # ==================================================

        "MOVE_UP": [

            "sube",
            "sube arriba",
            "ve arriba",
            "muévete arriba",
            "muevete arriba",
            "muévete hacia arriba",
            "muevete hacia arriba",
            "avanza arriba",
            "avanza hacia arriba",
            "camina arriba",
            "camina hacia arriba",
            "desplázate arriba",
            "desplazate arriba",
            "desplázate hacia arriba",
            "desplazate hacia arriba",
            "ve en dirección arriba",
            "ve en dirección hacia arriba",
            "continúa arriba",
            "continua arriba",
            "sigue arriba",
            "sigue hacia arriba",
            "quiero que subas",
            "quiero que vayas arriba",
            "haz que suba",
            "haz que el robot suba",
            "mueve el robot arriba",
            "mueve el robot hacia arriba",
            "lleva el robot arriba",
            "avanza en dirección arriba",
            "avanza en dirección hacia arriba",
            # Additional phrases to improve context distinction
            "asciende",
            "eleva tu posición",
            "muévete para arriba",
            "ve en sentido ascendente",
            "quiero que vayas hacia arriba",
            "dirígete a la parte superior",
            "dirigete a la parte superior",
            "sube un poco",
            "avanza hacia la parte de arriba",
            "escala"
        ],

        # ==================================================
        # MOVE DOWN
        # ==================================================

        "MOVE_DOWN": [

            "baja",
            "ve abajo",
            "muévete abajo",
            "muevete abajo",
            "muévete hacia abajo",
            "muevete hacia abajo",
            "avanza abajo",
            "avanza hacia abajo",
            "camina abajo",
            "camina hacia abajo",
            "desplázate abajo",
            "desplazate abajo",
            "desplázate hacia abajo",
            "desplazate hacia abajo",
            "ve en dirección abajo",
            "ve en dirección hacia abajo",
            "continúa abajo",
            "continua abajo",
            "sigue abajo",
            "sigue hacia abajo",
            "quiero que bajes",
            "quiero que vayas abajo",
            "haz que baje",
            "haz que el robot baje",
            "mueve el robot abajo",
            "mueve el robot hacia abajo",
            "lleva el robot abajo",
            "avanza en dirección abajo",
            "avanza en dirección hacia abajo",
            # Additional phrases to improve context distinction
            "desciende",
            "baja tu posición",
            "muévete para abajo",
            "ve en sentido descendente",
            "quiero que vayas hacia abajo",
            "dirígete a la parte inferior",
            "dirigete a la parte inferior",
            "baja un poco",
            "avanza hacia la parte de abajo",
            "desciende ahora"
        ],

        # ==================================================
        # MOVE LEFT
        # ==================================================

        "MOVE_LEFT": [

            "izquierda",
            "ve a la izquierda",
            "ve hacia la izquierda",
            "muévete a la izquierda",
            "muevete a la izquierda",
            "muévete hacia la izquierda",
            "muevete hacia la izquierda",
            "avanza a la izquierda",
            "avanza hacia la izquierda",
            "camina a la izquierda",
            "camina hacia la izquierda",
            "desplázate a la izquierda",
            "desplazate a la izquierda",
            "desplázate hacia la izquierda",
            "desplazate hacia la izquierda",
            "ve en dirección izquierda",
            "ve en dirección hacia la izquierda",
            "continúa a la izquierda",
            "continua a la izquierda",
            "sigue a la izquierda",
            "sigue hacia la izquierda",
            "quiero que vayas a la izquierda",
            "quiero que te muevas a la izquierda",
            "haz que vaya a la izquierda",
            "haz que el robot vaya a la izquierda",
            "mueve el robot a la izquierda",
            "mueve el robot hacia la izquierda",
            "lleva el robot a la izquierda",
            "avanza en dirección izquierda",
            "avanza en dirección hacia la izquierda",
            # Additional phrases for generalization
            "desplázate hacia el lado izquierdo",
            "desplazate hacia el lado izquierdo",
            "mueve la máquina hacia la izquierda",
            "quiero que el robot vaya hacia la izquierda",
            "gira a la izquierda",
            "ve al lado izquierdo",
            "dirígete al lado izquierdo"
        ],

        # ==================================================
        # MOVE RIGHT
        # ==================================================

        "MOVE_RIGHT": [

            "derecha",
            "ve a la derecha",
            "ve hacia la derecha",
            "muévete a la derecha",
            "muevete a la derecha",
            "muévete hacia la derecha",
            "muevete hacia la derecha",
            "avanza a la derecha",
            "avanza hacia la derecha",
            "camina a la derecha",
            "camina hacia la derecha",
            "desplázate a la derecha",
            "desplazate a la derecha",
            "desplázate hacia la derecha",
            "desplazate hacia la derecha",
            "ve en dirección derecha",
            "ve en dirección hacia la derecha",
            "continúa a la derecha",
            "continua a la derecha",
            "sigue a la derecha",
            "sigue hacia la derecha",
            "quiero que vayas a la derecha",
            "quiero que te muevas a la derecha",
            "haz que vaya a la derecha",
            "haz que el robot vaya a la derecha",
            "mueve el robot a la derecha",
            "mueve el robot hacia la derecha",
            "lleva el robot a la derecha",
            "avanza en dirección derecha",
            "avanza en dirección hacia la derecha",
            # Additional phrases for generalization
            "desplázate hacia el lado derecho",
            "desplazate hacia el lado derecho",
            "mueve la máquina hacia la derecha",
            "quiero que el robot vaya hacia la derecha",
            "gira a la derecha",
            "ve al lado derecho",
            "dirígete al lado derecho"
        ],

        # ==================================================
        # STOP
        # ==================================================

        "STOP": [

            "para",
            "detente",
            "detente ahora",
            "detente inmediatamente",
            "alto",
            "alto ahora",
            "para ahora",
            "deja de moverte",
            "deja de avanzar",
            "no te muevas",
            "quédate quieto",
            "quedate quieto",
            "quédate ahí",
            "quedate ahi",
            "espera",
            "espera ahí",
            "espera ahi",
            "haz una pausa",
            "detén el movimiento",
            "deten el movimiento",
            "detén al robot",
            "deten al robot",
            "para el robot",
            "detén el robot ahora",
            "deten el robot ahora",
            "quiero que pares",
            "quiero que te detengas",
            "haz que el robot se detenga",
            "deja quieto al robot",
            "no avances",
            # Additional phrases for generalization
            "quiero que el robot permanezca quieto",
            "permanece quieto",
            "quédate en tu lugar",
            "detén la máquina",
            "deten la maquina",
            "frena el robot",
            "cancela el movimiento",
            "frena ahora"
        ],
    }

    return data


def build_dataset(
    tokenizer,
    intent_vocab,
    train_ratio=0.8,
    seed=42,
):

    random.seed(seed)

    data = generate_language_data()

    train_sentences = []
    train_labels = []

    val_sentences = []
    val_labels = []

    for intent, sentences in data.items():

        sentences = list(set(sentences))

        random.shuffle(sentences)

        split_index = int(
            len(sentences) * train_ratio
        )

        train_data = sentences[:split_index]
        val_data = sentences[split_index:]

        label = intent_vocab.encode(intent)

        for sentence in train_data:
            train_sentences.append(sentence)
            train_labels.append(label)

        for sentence in val_data:
            val_sentences.append(sentence)
            val_labels.append(label)

    train_combined = list(
        zip(
            train_sentences,
            train_labels,
        )
    )

    val_combined = list(
        zip(
            val_sentences,
            val_labels,
        )
    )

    random.shuffle(train_combined)
    random.shuffle(val_combined)

    train_sentences, train_labels = zip(
        *train_combined
    )

    val_sentences, val_labels = zip(
        *val_combined
    )

    return (
        list(train_sentences),
        list(train_labels),
        list(val_sentences),
        list(val_labels),
    )