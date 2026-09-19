import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rbj.language.vocabulary import Vocabulary
from rbj.language.tokenizer import Tokenizer
from rbj.language.dataset import (
    LanguageDataset,
    IntentVocabulary,
    build_dataset,
)
from rbj.language.model import SJGLanguage


def main():

    print("=" * 60)
    print("SJG-AGENT - LANGUAGE MODEL 0.3.3")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Vocabulary
    # --------------------------------------------------

    vocab = Vocabulary()

    tokenizer = Tokenizer(vocab)

    # --------------------------------------------------
    # 2. Intent vocabulary
    # --------------------------------------------------

    intent_vocab = IntentVocabulary()

    # --------------------------------------------------
    # 3. Generar dataset
    # --------------------------------------------------

    (
        train_sentences,
        train_labels,
        val_sentences,
        val_labels,
    ) = build_dataset(
        tokenizer=tokenizer,
        intent_vocab=intent_vocab,
        train_ratio=0.8,
        seed=42,
    )

    # --------------------------------------------------
    # 4. Construir vocabulario
    # --------------------------------------------------

    all_sentences = (
        train_sentences +
        val_sentences
    )

    for sentence in all_sentences:
        vocab.add_sentence(sentence)

    print()
    print("Vocabulario:", len(vocab))
    print("Clases:", len(intent_vocab))

    # --------------------------------------------------
    # 5. Crear datasets
    # --------------------------------------------------

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

    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Max length:", train_dataset.max_length)

    # --------------------------------------------------
    # 6. Crear modelo
    # --------------------------------------------------

    model = SJGLanguage(
        vocab_size=len(vocab),
        embedding_dim=64,
        hidden_dim=128,
        num_classes=len(intent_vocab),
        padding_idx=vocab.word_to_id["<PAD>"],
    )

    print()
    print("MODELO")
    print("-" * 60)

    print(model)

    # --------------------------------------------------
    # 7. Tomar una muestra
    # --------------------------------------------------

    sample = train_dataset[0]

    input_ids = sample["input_ids"].unsqueeze(0)

    print()
    print("INPUT")
    print("-" * 60)

    print("Shape:", input_ids.shape)
    print("IDs:", input_ids.tolist())

    # --------------------------------------------------
    # 8. Forward pass
    # --------------------------------------------------

    logits = model(input_ids)

    print()
    print("OUTPUT")
    print("-" * 60)

    print("Shape:", logits.shape)
    print("Logits:", logits.tolist())

    # --------------------------------------------------
    # 9. Predicción
    # --------------------------------------------------

    prediction = model.predict(input_ids)

    prediction_id = prediction.item()

    prediction_name = intent_vocab.decode(
        prediction_id
    )

    real_id = sample["label"].item()

    real_name = intent_vocab.decode(
        real_id
    )

    print()
    print("PREDICCIÓN")
    print("-" * 60)

    print("Texto:", train_sentences[0])
    print("Real:", real_name)
    print("Predicción:", prediction_name)

    print()
    print("=" * 60)
    print("MODEL OK")
    print("=" * 60)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print("Parámetros totales:", total_parameters)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Parámetros entrenables:",
        trainable_parameters
    )


if __name__ == "__main__":
    main()