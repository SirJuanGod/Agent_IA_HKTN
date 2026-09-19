class Vocabulary:

    PAD = "<PAD>"
    UNK = "<UNK>"
    START = "<START>"
    END = "<END>"

    def __init__(self):

        self.word_to_id = {
            self.PAD: 0,
            self.UNK: 1,
            self.START: 2,
            self.END: 3,
        }

        self.id_to_word = {
            0: self.PAD,
            1: self.UNK,
            2: self.START,
            3: self.END,
        }

    # --------------------------------------------------
    # Propiedades de acceso rápido
    # --------------------------------------------------

    @property
    def pad_id(self):
        return self.word_to_id[self.PAD]

    @property
    def unk_id(self):
        return self.word_to_id[self.UNK]

    @property
    def start_id(self):
        return self.word_to_id[self.START]

    @property
    def end_id(self):
        return self.word_to_id[self.END]

    def add_word(self, word):

        if word not in self.word_to_id:

            index = len(self.word_to_id)

            self.word_to_id[word] = index

            self.id_to_word[index] = word

        return self.word_to_id[word]

    def add_sentence(self, sentence):

        words = sentence.split()

        for word in words:

            self.add_word(word)

    def encode_word(self, word):

        return self.word_to_id.get(
            word,
            self.unk_id
        )

    def decode_id(self, index):

        return self.id_to_word.get(
            index,
            self.UNK
        )

    def __len__(self):

        return len(self.word_to_id)

    def save(self, path):

        import json

        data = {
            "word_to_id": self.word_to_id,
            "id_to_word": self.id_to_word
        }

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=4
            )

    def load(self, path):

        import json

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        self.word_to_id = data["word_to_id"]

        self.id_to_word = {
            int(k): v
            for k, v in data["id_to_word"].items()
        }