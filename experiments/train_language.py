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
import torch.nn as nn
from torch.utils.data import DataLoader

from rbj.language.vocabulary import Vocabulary
from rbj.language.tokenizer import Tokenizer
from rbj.language.dataset import (
    LanguageDataset,
    IntentVocabulary,
    build_dataset,
)
from rbj.language.model import SJGLanguage


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

SEED = 42

BATCH_SIZE = 24
EPOCHS = 100

LEARNING_RATE = 0.0005

EMBEDDING_DIM = 64
HIDDEN_DIM = 128

TRAIN_RATIO = 0.9

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "sjg_language_036.pt"
)


# ==========================================================
# SEED
# ==========================================================

torch.manual_seed(SEED)


# ==========================================================
# EVALUACIÓN
# ==========================================================

def evaluate(model, dataloader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)

            loss = criterion(
                logits,
                labels
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = total_loss / len(dataloader)

    accuracy = correct / total

    return average_loss, accuracy


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 60)
    print("SJG-AGENT - LANGUAGE TRAINING 0.3.4")
    print("=" * 60)

    # ------------------------------------------------------
    # Device
    # ------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    # ------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------

    vocab = Vocabulary()

    tokenizer = Tokenizer(vocab)

    # ------------------------------------------------------
    # Intent vocabulary
    # ------------------------------------------------------

    intent_vocab = IntentVocabulary()

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------

    (
        train_sentences,
        train_labels,
        val_sentences,
        val_labels,
    ) = build_dataset(
        tokenizer=tokenizer,
        intent_vocab=intent_vocab,
        train_ratio=TRAIN_RATIO,
        seed=SEED,
    )

    # ------------------------------------------------------
    # Construir vocabulario
    # ------------------------------------------------------

    all_sentences = (
        train_sentences +
        val_sentences
    )

    for sentence in all_sentences:
        vocab.add_sentence(sentence)

    print()
    print("Vocabulary size:", len(vocab))
    print("Number of classes:", len(intent_vocab))

    # ------------------------------------------------------
    # Datasets
    # ------------------------------------------------------

    train_dataset = LanguageDataset(
        sentences=train_sentences,
        labels=train_labels,
        tokenizer=tokenizer,
    )

    val_dataset = LanguageDataset(
        sentences=val_sentences,
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=train_dataset.max_length,
    )

    print()
    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Sequence length:", train_dataset.max_length)

    # ------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=0,
    )

    # ------------------------------------------------------
    # Modelo
    # ------------------------------------------------------

    model = SJGLanguage(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=len(intent_vocab),
        padding_idx=vocab.word_to_id["<PAD>"],
    )

    model = model.to(device)

    # ------------------------------------------------------
    # Parámetros
    # ------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print("Model parameters:", total_parameters)

    # ------------------------------------------------------
    # Loss
    # ------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------

    print()
    print("-" * 60)
    print("TRAINING")
    print("-" * 60)

    best_val_accuracy = 0.0

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:

            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            # ----------------------------------------------
            # Forward
            # ----------------------------------------------

            logits = model(input_ids)

            # ----------------------------------------------
            # Loss
            # ----------------------------------------------

            loss = criterion(
                logits,
                labels
            )

            # ----------------------------------------------
            # Backpropagation
            # ----------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            # ----------------------------------------------
            # Update
            # ----------------------------------------------

            optimizer.step()

            # ----------------------------------------------
            # Metrics
            # ----------------------------------------------

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = (
            total_loss /
            len(train_loader)
        )

        train_accuracy = (
            correct /
            total
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        # --------------------------------------------------
        # Output
        # --------------------------------------------------

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.3f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.3f}"
        )

        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            os.makedirs(
                CHECKPOINT_DIR,
                exist_ok=True
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "vocab_state":
                        vocab.word_to_id,

                    "intent_state":
                        intent_vocab.word_to_id,

                    "embedding_dim":
                        EMBEDDING_DIM,

                    "hidden_dim":
                        HIDDEN_DIM,

                    "max_length":
                        train_dataset.max_length,

                    "val_accuracy":
                        val_accuracy,
                },
                CHECKPOINT_PATH,
            )

            print(
                f"  ✓ Checkpoint guardado: "
                f"{CHECKPOINT_PATH}"
            )

    # ------------------------------------------------------
    # Final
    # ------------------------------------------------------

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.3f}"
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )


if __name__ == "__main__":
    main()