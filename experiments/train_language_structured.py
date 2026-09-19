import json
from pathlib import Path
import sys
sys.path.append(
    str(
        Path(__file__).resolve().parent.parent
    )
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rbj.language.vocabulary import Vocabulary
from rbj.language.tokenizer import Tokenizer

from rbj.language.structured_dataset import (
    StructuredLanguageDataset,
    build_label_maps,
)

from rbj.language.structured_model import (
    SJGLanguageStructured,
)


BATCH_SIZE = 32
EPOCHS = 80

LEARNING_RATE = 0.0005

MAX_LENGTH = 24

EMBEDDING_DIM = 96
HIDDEN_DIM = 192

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def main():

    print("=" * 70)
    print("SJG-Agent")
    print("0.3.7 - Structured Language Training")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    train_data = load_json(
        "dataset/language/train.json"
    )

    validation_data = load_json(
        "dataset/language/validation.json"
    )

    # -----------------------------------------------------
    # Vocabulary
    # -----------------------------------------------------

    vocab = Vocabulary()

    tokenizer = Tokenizer(vocab)

    for item in train_data:

        for token in tokenizer.tokenize(
            item["text"]
        ):

            vocab.add_word(token)

    # -----------------------------------------------------
    # Label maps
    # -----------------------------------------------------

    label_maps = build_label_maps(
        train_data
    )

    print(
        f"Vocabulary: {len(vocab)}"
    )

    for field, mapping in label_maps.items():

        print(
            f"{field}: {mapping}"
        )

    # -----------------------------------------------------
    # Dataset objects
    # -----------------------------------------------------

    train_dataset = StructuredLanguageDataset(
        train_data,
        tokenizer,
        label_maps,
        MAX_LENGTH,
    )

    validation_dataset = StructuredLanguageDataset(
        validation_data,
        tokenizer,
        label_maps,
        MAX_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = SJGLanguageStructured(
        vocab_size=len(vocab),

        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,

        num_intents=len(
            label_maps["intent"]
        ),

        num_actions=len(
            label_maps["action"]
        ),

        num_directions=len(
            label_maps["direction"]
        ),

        num_targets=len(
            label_maps["target"]
        ),

        num_subjects=len(
            label_maps["subject"]
        ),

        max_distance=4,
    ).to(DEVICE)

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    best_score = 0.0

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        train_loss = 0.0

        for inputs, labels in train_loader:

            inputs = inputs.to(
                DEVICE
            )

            labels = {
                key: value.to(DEVICE)
                for key, value in labels.items()
            }

            optimizer.zero_grad()

            outputs = model(
                inputs
            )

            loss = (

                criterion(
                    outputs["intent"],
                    labels["intent"]
                )

                + criterion(
                    outputs["action"],
                    labels["action"]
                )

                + criterion(
                    outputs["direction"],
                    labels["direction"]
                )

                + criterion(
                    outputs["target"],
                    labels["target"]
                )

                + criterion(
                    outputs["subject"],
                    labels["subject"]
                )

                + criterion(
                    outputs["distance"],
                    labels["distance"]
                )
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        model.eval()

        correct = {
            "intent": 0,
            "action": 0,
            "direction": 0,
            "target": 0,
            "subject": 0,
            "distance": 0,
        }

        total = 0

        with torch.no_grad():

            for inputs, labels in validation_loader:

                inputs = inputs.to(
                    DEVICE
                )

                labels = {
                    key: value.to(DEVICE)
                    for key, value in labels.items()
                }

                outputs = model(
                    inputs
                )

                batch_size = (
                    inputs.size(0)
                )

                total += batch_size

                for field in correct:

                    predictions = (
                        outputs[field]
                        .argmax(dim=1)
                    )

                    correct[field] += (
                        predictions
                        == labels[field]
                    ).sum().item()

        accuracies = {
            field:
                correct[field] / total
            for field in correct
        }

        # Exact match:
        # todas las salidas correctas
        exact = 0

        with torch.no_grad():

            for inputs, labels in validation_loader:

                inputs = inputs.to(
                    DEVICE
                )

                labels = {
                    key: value.to(DEVICE)
                    for key, value in labels.items()
                }

                outputs = model(
                    inputs
                )

                predictions = {
                    field:
                        outputs[field]
                        .argmax(dim=1)

                    for field in labels
                }

                batch_correct = torch.ones(
                    inputs.size(0),
                    dtype=torch.bool,
                    device=DEVICE
                )

                for field in labels:

                    batch_correct &= (
                        predictions[field]
                        == labels[field]
                    )

                exact += (
                    batch_correct
                    .sum()
                    .item()
                )

        exact_accuracy = (
            exact / total
        )

        avg_accuracy = sum(
            accuracies.values()
        ) / len(
            accuracies
        )

        print(
            f"\nEpoch {epoch:03d}"
        )

        print(
            f"Loss: {train_loss:.4f}"
        )

        print(
            f"Intent:     {accuracies['intent']:.3f}"
        )

        print(
            f"Action:     {accuracies['action']:.3f}"
        )

        print(
            f"Direction:  {accuracies['direction']:.3f}"
        )

        print(
            f"Target:     {accuracies['target']:.3f}"
        )

        print(
            f"Subject:    {accuracies['subject']:.3f}"
        )

        print(
            f"Distance:   {accuracies['distance']:.3f}"
        )

        print(
            f"Average:    {avg_accuracy:.3f}"
        )

        print(
            f"Exact:      {exact_accuracy:.3f}"
        )

        # -------------------------------------------------
        # Checkpoint
        # -------------------------------------------------

        if exact_accuracy > best_score:

            best_score = exact_accuracy

            checkpoint = {

                "model_state_dict":
                    model.state_dict(),

                "vocab_state":
                    vocab.word_to_id,

                "label_maps":
                    label_maps,

                "embedding_dim":
                    EMBEDDING_DIM,

                "hidden_dim":
                    HIDDEN_DIM,

                "max_length":
                    MAX_LENGTH,

                "max_distance":
                    4,

                "exact_accuracy":
                    exact_accuracy,
            }

            output = Path(
                "checkpoints/"
                "sjg_language_037.pt"
            )

            output.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                checkpoint,
                output
            )

            print(
                f"Checkpoint guardado: {output}"
            )


if __name__ == "__main__":
    main()