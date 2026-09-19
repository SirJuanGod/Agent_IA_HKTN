import re


class Tokenizer:

    def __init__(self, vocabulary):

        self.vocabulary = vocabulary

    def normalize(self, text):

        text = text.lower()

        text = text.strip()

        # Separar puntuación
        text = re.sub(
            r"([.,!?;:])",
            r" \1 ",
            text
        )

        # Eliminar espacios repetidos
        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text

    def tokenize(self, text):

        text = self.normalize(text)

        return text.split()

    def encode(
        self,
        text,
        add_start=True,
        add_end=True
    ):

        tokens = self.tokenize(text)

        ids = []

        if add_start:

            ids.append(
                self.vocabulary.encode_word(
                    self.vocabulary.START
                )
            )

        for token in tokens:

            ids.append(
                self.vocabulary.encode_word(
                    token
                )
            )

        if add_end:

            ids.append(
                self.vocabulary.encode_word(
                    self.vocabulary.END
                )
            )

        return ids

    def decode(
        self,
        ids,
        remove_special=True
    ):

        words = []

        for index in ids:

            word = self.vocabulary.decode_id(
                int(index)
            )

            if remove_special:

                if word in [
                    self.vocabulary.PAD,
                    self.vocabulary.START,
                    self.vocabulary.END
                ]:

                    continue

            words.append(word)

        return " ".join(words)