import os
import sys

# Permite ejecutar este módulo directamente desde la raíz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rbj.language.vocabulary import Vocabulary
from rbj.language.tokenizer import Tokenizer
from rbj.language.dataset import (
    LanguageDataset,
    IntentVocabulary,
    build_dataset,
)


def main():

    print("=" * 50)
    print("SJG-AGENT - LANGUAGE DATASET 0.3.2")
    print("=" * 50)

    # --------------------------------------------------
    # 1. Crear vocabulario
    # --------------------------------------------------

    vocab = Vocabulary()

    tokenizer = Tokenizer(vocab)

    # --------------------------------------------------
    # 2. Generar datos
    # --------------------------------------------------

    intent_vocab = IntentVocabulary()

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
    # 3. Agregar frases al vocabulario
    # --------------------------------------------------

    # Necesitamos construir el vocabulario antes
    # de crear el Dataset.

    all_sentences = (
        train_sentences +
        val_sentences
    )

    for sentence in all_sentences:
        vocab.add_sentence(sentence)

    # --------------------------------------------------
    # 4. Crear datasets
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

    # --------------------------------------------------
    # 5. Información
    # --------------------------------------------------

    print()
    print("Número de clases:", len(intent_vocab))
    print("Tamaño vocabulario:", len(vocab))

    print()
    print("Dataset entrenamiento:")
    print(len(train_dataset))

    print()
    print("Dataset validación:")
    print(len(val_dataset))

    print()
    print("Longitud máxima:")
    print(train_dataset.max_length)

    # --------------------------------------------------
    # 6. Mostrar muestras
    # --------------------------------------------------

    print()
    print("MUESTRAS")
    print("-" * 50)

    for i in range(min(10, len(train_dataset))):

        sample = train_dataset[i]

        sentence = train_sentences[i]
        label_id = sample["label"].item()
        label = intent_vocab.decode(label_id)

        print()
        print("Texto :", sentence)
        print("Tokens:", sample["input_ids"].tolist())
        print("Clase :", label)
        print("ID    :", label_id)

    # --------------------------------------------------
    # 7. Prueba de reconstrucción
    # --------------------------------------------------

    print()
    print("RECONSTRUCCIÓN")
    print("-" * 50)

    sample_sentence = train_sentences[0]

    encoded = tokenizer.encode(
        sample_sentence
    )

    decoded = tokenizer.decode(
        encoded
    )

    print("Original :", sample_sentence)
    print("Encoded  :", encoded)
    print("Decoded  :", decoded)

    print()
    print("=" * 50)
    print("DATASET OK")
    print("=" * 50)


if __name__ == "__main__":
    main()