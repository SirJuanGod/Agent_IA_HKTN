import os
import sys
import warnings

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
from torch.utils.data import DataLoader

from rbj.language.vocabulary import Vocabulary
from rbj.language.tokenizer import Tokenizer
from rbj.language.dataset import (
    LanguageDataset,
    IntentVocabulary,
    build_dataset,
)
from rbj.language.model import SJGLanguage


CHECKPOINT = "checkpoints/sjg_language_036.pt"

SEED = 42


def main():

    print("=" * 60)
    print("SJG-AGENT - LANGUAGE EVALUATION 0.3.6")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------
    # Checkpoint
    # --------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    # --------------------------------------------------
    # Vocabulary
    # --------------------------------------------------

    vocab = Vocabulary()

    vocab.word_to_id = checkpoint[
        "vocab_state"
    ]

    vocab.id_to_word = {
        idx: word
        for word, idx
        in vocab.word_to_id.items()
    }

    tokenizer = Tokenizer(vocab)

    # --------------------------------------------------
    # Intent vocabulary
    # --------------------------------------------------

    intent_vocab = IntentVocabulary()

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    (
        train_sentences,
        train_labels,
        val_sentences,
        val_labels,
    ) = build_dataset(
        tokenizer,
        intent_vocab,
        train_ratio=0.8,
        seed=SEED,
    )

    val_dataset = LanguageDataset(
        sentences=val_sentences,
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=checkpoint["max_length"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
    )

    # --------------------------------------------------
    # Modelo
    # --------------------------------------------------

    model = SJGLanguage(
        vocab_size=len(vocab),
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=len(intent_vocab),
        padding_idx=vocab.word_to_id["<PAD>"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    num_classes = len(intent_vocab)

    confusion_matrix = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.int64,
    )

    with torch.no_grad():

        for batch in val_loader:

            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            for real, predicted in zip(
                labels,
                predictions,
            ):

                confusion_matrix[
                    real.item(),
                    predicted.item(),
                ] += 1

    # --------------------------------------------------
    # Resultados
    # --------------------------------------------------

    print()
    print("VALIDATION RESULTS")
    print("-" * 60)

    total_correct = 0
    total_samples = 0

    for class_id in range(num_classes):

        class_total = (
            confusion_matrix[class_id]
            .sum()
            .item()
        )

        class_correct = (
            confusion_matrix[
                class_id,
                class_id
            ].item()
        )

        if class_total > 0:

            accuracy = (
                class_correct /
                class_total
            )

        else:

            accuracy = 0.0

        total_correct += class_correct
        total_samples += class_total

        intent = intent_vocab.decode(
            class_id
        )

        print(
            f"{intent:15s} "
            f"{class_correct:3d}/"
            f"{class_total:3d} "
            f"→ "
            f"{accuracy:.3f}"
        )

    # --------------------------------------------------
    # Global accuracy
    # --------------------------------------------------

    global_accuracy = (
        total_correct /
        total_samples
    )

    print()
    print(
        f"Global accuracy: "
        f"{global_accuracy:.3f}"
    )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    print()
    print("CONFUSION MATRIX")
    print("-" * 60)

    print(
        "Rows = real | Columns = predicted"
    )

    print()

    header = " " * 15

    for i in range(num_classes):

        header += (
            f"{i:^12}"
        )

    print(header)

    for i in range(num_classes):

        row = (
            f"{intent_vocab.decode(i):15s}"
        )

        for j in range(num_classes):

            row += (
                f"{confusion_matrix[i, j].item():^12}"
            )

        print(row)

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()