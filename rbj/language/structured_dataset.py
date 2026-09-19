import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class LanguageExample:
    id: str
    text: str

    intent: str
    action: str

    direction: str
    target: str
    subject: str

    distance: int

    mode: str
    difficulty: int

    template_family: str


class StructuredLanguageGenerator:

    DIRECTIONS = {
        "UP": [
            "arriba",
            "hacia arriba",
            "la parte superior",
            "el lado de arriba",
        ],
        "DOWN": [
            "abajo",
            "hacia abajo",
            "la parte inferior",
            "el lado de abajo",
        ],
        "LEFT": [
            "a la izquierda",
            "hacia la izquierda",
            "el lado izquierdo",
            "la parte izquierda",
        ],
        "RIGHT": [
            "a la derecha",
            "hacia la derecha",
            "el lado derecho",
            "la parte derecha",
        ],
    }

    SUBJECTS = [
        "el robot",
        "el agente",
        "la máquina",
        "mi robot",
    ]

    GOALS = [
        "la meta",
        "el objetivo",
        "el destino",
        "el punto objetivo",
    ]

    DISTANCES = {
        1: [
            "un paso",
            "una casilla",
            "una posición",
        ],
        2: [
            "dos pasos",
            "dos casillas",
            "dos posiciones",
        ],
        3: [
            "tres pasos",
            "tres casillas",
            "tres posiciones",
        ],
        4: [
            "cuatro pasos",
            "cuatro casillas",
            "cuatro posiciones",
        ],
    }

    def __init__(self, seed=42):

        self.random = random.Random(seed)
        self.counter = 0

    # =========================================================
    # ID
    # =========================================================

    def _id(self):

        self.counter += 1

        return f"lang_{self.counter:06d}"

    # =========================================================
    # MOVE
    # =========================================================

    def generate_move(self, direction, distance):

        examples = []

        d = self.random.choice(
            self.DIRECTIONS[direction]
        )

        subject = self.random.choice(
            self.SUBJECTS
        )

        distance_text = self.random.choice(
            self.DISTANCES[distance]
        )

        # -----------------------------------------------------
        # Familia 1: orden directa
        # -----------------------------------------------------

        direct_templates = [
            f"ve {d}",
            f"muévete {d}",
            f"avanza {d}",
            f"dirígete {d}",
        ]

        for text in direct_templates:

            examples.append(
                self._move_example(
                    text=text,
                    direction=direction,
                    distance=1,
                    difficulty=1,
                    family="direct",
                )
            )

        # -----------------------------------------------------
        # Familia 2: imperativo
        # -----------------------------------------------------

        imperative_templates = [
            f"sube" if direction == "UP" else None,
            f"baja" if direction == "DOWN" else None,
            f"ve a la izquierda"
            if direction == "LEFT"
            else None,
            f"ve a la derecha"
            if direction == "RIGHT"
            else None,
        ]

        for text in imperative_templates:

            if text is None:
                continue

            examples.append(
                self._move_example(
                    text=text,
                    direction=direction,
                    distance=1,
                    difficulty=1,
                    family="imperative",
                )
            )

        # -----------------------------------------------------
        # Familia 3: sujeto
        # -----------------------------------------------------

        subject_templates = [
            f"{subject} debe ir {d}",
            f"{subject} debe moverse {d}",
            f"{subject} tiene que avanzar {d}",
            f"haz que {subject} vaya {d}",
            f"quiero que {subject} se mueva {d}",
        ]

        for text in subject_templates:

            examples.append(
                self._move_example(
                    text=text,
                    direction=direction,
                    distance=1,
                    difficulty=2,
                    family="subject",
                )
            )

        # -----------------------------------------------------
        # Familia 4: distancia
        # -----------------------------------------------------

        distance_templates = [
            f"avanza {distance_text} hacia {d}",
            f"muévete {distance_text} hacia {d}",
            f"recorre {distance_text} hacia {d}",
            f"desplázate {distance_text} hacia {d}",
        ]

        for text in distance_templates:

            examples.append(
                self._move_example(
                    text=text,
                    direction=direction,
                    distance=distance,
                    difficulty=3,
                    family="distance",
                )
            )

        # -----------------------------------------------------
        # Familia 5: sujeto + distancia
        # -----------------------------------------------------

        subject_distance_templates = [
            (
                f"haz que {subject} avance "
                f"{distance_text} hacia {d}"
            ),
            (
                f"quiero que {subject} se mueva "
                f"{distance_text} hacia {d}"
            ),
            (
                f"{subject} debe avanzar "
                f"{distance_text} hacia {d}"
            ),
            (
                f"{subject} tiene que recorrer "
                f"{distance_text} hacia {d}"
            ),
        ]

        for text in subject_distance_templates:

            examples.append(
                self._move_example(
                    text=text,
                    direction=direction,
                    distance=distance,
                    difficulty=4,
                    family="subject_distance",
                )
            )

        # -----------------------------------------------------
        # Familia 6: lenguaje natural
        # -----------------------------------------------------

        natural_templates = [
            (
                f"quiero que {subject} "
                f"se desplace hacia {d}"
            ),
            (
                f"necesito que {subject} "
                f"avance hacia {d}"
            ),
            (
                f"haz avanzar {subject} "
                f"hacia {d}"
            ),
            (
                f"lleva {subject} hacia {d}"
            ),
        ]

        for text in natural_templates:

            examples.append(
                self._move_example(
                    text=text,
                    direction=direction,
                    distance=1,
                    difficulty=4,
                    family="natural",
                )
            )

        return examples

    # =========================================================
    # MOVE EXAMPLE
    # =========================================================

    def _move_example(
        self,
        text,
        direction,
        distance,
        difficulty,
        family,
    ):

        return LanguageExample(
            id=self._id(),
            text=text,
            intent="MOVE",
            action="MOVE",
            direction=direction,
            target="NONE",
            subject="ROBOT",
            distance=distance,
            mode="COMMAND",
            difficulty=difficulty,
            template_family=family,
        )

    # =========================================================
    # GO TO
    # =========================================================

    def generate_go_to(self):

        examples = []

        goal = self.random.choice(
            self.GOALS
        )

        subject = self.random.choice(
            self.SUBJECTS
        )

        # -----------------------------------------------------
        # Direct
        # -----------------------------------------------------

        direct = [
            f"ve hasta {goal}",
            f"ve hacia {goal}",
            f"llega hasta {goal}",
            f"llega a {goal}",
            f"dirígete a {goal}",
            f"dirígete hasta {goal}",
        ]

        for text in direct:

            examples.append(
                self._goto_example(
                    text=text,
                    difficulty=1,
                    family="goto_direct",
                )
            )

        # -----------------------------------------------------
        # Subject
        # -----------------------------------------------------

        subject_templates = [
            f"{subject} debe ir hasta {goal}",
            f"{subject} tiene que llegar a {goal}",
            f"haz que {subject} llegue a {goal}",
            f"quiero que {subject} vaya hasta {goal}",
        ]

        for text in subject_templates:

            examples.append(
                self._goto_example(
                    text=text,
                    difficulty=2,
                    family="goto_subject",
                )
            )

        # -----------------------------------------------------
        # Natural
        # -----------------------------------------------------

        natural = [
            f"lleva {subject} hasta {goal}",
            f"haz llegar {subject} a {goal}",
            f"quiero llevar el robot hasta {goal}",
            f"necesito que el robot alcance {goal}",
        ]

        for text in natural:

            examples.append(
                self._goto_example(
                    text=text,
                    difficulty=4,
                    family="goto_natural",
                )
            )

        return examples

    # =========================================================
    # GO TO EXAMPLE
    # =========================================================

    def _goto_example(
        self,
        text,
        difficulty,
        family,
    ):

        return LanguageExample(
            id=self._id(),
            text=text,
            intent="GO_TO",
            action="GO_TO",
            direction="NONE",
            target="GOAL",
            subject="ROBOT",
            distance=0,
            mode="COMMAND",
            difficulty=difficulty,
            template_family=family,
        )

    # =========================================================
    # STOP
    # =========================================================

    def generate_stop(self):

        phrases = [
            ("detente", "stop_direct"),
            ("para", "stop_direct"),
            ("alto", "stop_direct"),
            ("quédate quieto", "stop_direct"),

            ("detén el robot", "stop_subject"),
            ("haz que el robot se detenga", "stop_subject"),
            ("quiero que el robot se detenga", "stop_subject"),

            ("deja de moverte", "stop_natural"),
            ("deja de avanzar", "stop_natural"),
            ("no te muevas", "stop_natural"),
            ("mantente quieto", "stop_natural"),
        ]

        examples = []

        for text, family in phrases:

            difficulty = 1

            if family == "stop_subject":
                difficulty = 2

            elif family == "stop_natural":
                difficulty = 3

            examples.append(
                LanguageExample(
                    id=self._id(),
                    text=text,
                    intent="STOP",
                    action="STOP",
                    direction="NONE",
                    target="NONE",
                    subject="ROBOT",
                    distance=0,
                    mode="COMMAND",
                    difficulty=difficulty,
                    template_family=family,
                )
            )

        return examples

    # =========================================================
    # GENERACIÓN COMPLETA
    # =========================================================

    def generate_all(self):

        examples = []

        for direction in [
            "UP",
            "DOWN",
            "LEFT",
            "RIGHT",
        ]:

            for distance in [
                1,
                2,
                3,
                4,
            ]:

                examples.extend(
                    self.generate_move(
                        direction,
                        distance
                    )
                )

        for _ in range(20):

            examples.extend(
                self.generate_go_to()
            )

        examples.extend(
            self.generate_stop()
        )

        return examples

    # =========================================================
    # SPLIT POR FAMILIA
    # =========================================================

    def split_dataset(self, examples):

        train_families = {
            "direct",
            "imperative",
            "subject",
            "distance",
            "goto_direct",
            "goto_subject",
            "stop_direct",
            "stop_subject",
        }

        validation_families = {
            "subject_distance",
            "goto_natural",
            "stop_natural",
        }

        test_families = {
            "natural",
        }

        train = []
        validation = []
        test = []

        for example in examples:

            family = example.template_family

            if family in train_families:

                train.append(example)

            elif family in validation_families:

                validation.append(example)

            elif family in test_families:

                test.append(example)

        self.random.shuffle(train)
        self.random.shuffle(validation)
        self.random.shuffle(test)

        return train, validation, test

    # =========================================================
    # AMBIGUOUS
    # =========================================================

    def generate_ambiguous(self):

        phrases = [
            "avanza",
            "muévete",
            "ve",
            "sigue",
            "adelante",
            "muévete un poco",
            "avanza un poco",
            "ve por ese lado",
            "muévete por ahí",
            "sigue por ese camino",
            "ve por allá",
            "muévete hacia ese lado",
            "haz algo",
            "continúa",
            "sigue avanzando",
        ]

        return [
            {
                "id": f"ambiguous_{i:04d}",
                "text": text,
                "expected": "AMBIGUOUS",
                "reason": self._ambiguity_reason(text),
            }
            for i, text in enumerate(
                phrases,
                start=1
            )
        ]

    @staticmethod
    def _ambiguity_reason(text):

        if text in {
            "avanza",
            "muévete",
            "ve",
            "sigue",
            "adelante",
            "continúa",
            "sigue avanzando",
        }:

            return "MISSING_DIRECTION"

        if "un poco" in text:

            return "MISSING_DIRECTION"

        if (
            "ese lado" in text
            or "por ahí" in text
            or "por allá" in text
            or "ese camino" in text
        ):

            return "UNRESOLVED_REFERENCE"

        return "INSUFFICIENT_INFORMATION"


# =============================================================
# SAVE
# =============================================================

def save_json(data, path):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


def save_examples(examples, path):

    data = [
        asdict(example)
        for example in examples
    ]

    save_json(
        data,
        path
    )


class StructuredLanguageDataset(Dataset):

    def __init__(
        self,
        data,
        tokenizer,
        label_maps,
        max_length=20,
    ):

        self.data = data
        self.tokenizer = tokenizer
        self.label_maps = label_maps
        self.max_length = max_length

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        item = self.data[index]

        token_ids = self.tokenizer.encode(
            item["text"]
        )

        # ---------------------------------------------
        # Padding
        # ---------------------------------------------

        if len(token_ids) < self.max_length:

            token_ids += [
                self.tokenizer.vocabulary.pad_id
            ] * (
                self.max_length - len(token_ids)
            )

        else:

            token_ids = token_ids[
                :self.max_length
            ]

        # ---------------------------------------------
        # Labels
        # ---------------------------------------------

        labels = {

            "intent": self.label_maps[
                "intent"
            ][item["intent"]],

            "action": self.label_maps[
                "action"
            ][item["action"]],

            "direction": self.label_maps[
                "direction"
            ][item["direction"]],

            "target": self.label_maps[
                "target"
            ][item["target"]],

            "subject": self.label_maps[
                "subject"
            ][item["subject"]],

            "distance": item["distance"],
        }

        return (
            torch.tensor(
                token_ids,
                dtype=torch.long
            ),
            {
                key: torch.tensor(
                    value,
                    dtype=torch.long
                )
                for key, value
                in labels.items()
            }
        )


def build_label_maps(data):

    fields = [
        "intent",
        "action",
        "direction",
        "target",
        "subject",
    ]

    maps = {}

    for field in fields:

        values = sorted(
            set(
                item[field]
                for item in data
            )
        )

        maps[field] = {
            value: index
            for index, value
            in enumerate(values)
        }

    return maps