# src/data_loader/tokenizer.py
import os


class RecipeTasteTokenizer:
    def __init__(
        self, ingredients_path="config/ingredients.txt", states_path="config/states.txt"
    ):
        # 特殊トークンの定義
        self.PAD_TOKEN = "[PAD]"
        self.MASK_TOKEN = "[MASK]"
        self.UNK_TOKEN = "[UNK]"

        # 1. 材料の辞書（Vocab）
        self.ing_to_id = {}
        self.id_to_ing = {}

        # 2. 状態の辞書（Vocab）
        self.state_to_id = {}
        self.id_to_state = {}

        # 初期設定として、最低限の単語を登録（あとからファイル読み込みに拡張可能）
        self._load_vocab_from_file(ingredients_path, self.ing_to_id, self.id_to_ing)
        self._load_vocab_from_file(states_path, self.state_to_id, self.id_to_state)

    def _load_vocab_from_file(self, file_path, to_id_dict, to_char_dict):
        if not os.path.exists(file_path):
            raise FileNotFoundError("no file founded")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word and (word not in to_id_dict):
                    to_id_dict[word] = len(to_id_dict)

        for k, v in to_id_dict.items():
            to_char_dict[v] = k

    def encode_ingredient(self, text):
        """材料テキストをIDに変換（未知語は[UNK]）"""
        return self.ing_to_id.get(text, self.ing_to_id[self.UNK_TOKEN])

    def encode_state(self, text):
        """状態テキストをIDに変換（未知語は[UNK]）"""
        return self.state_to_id.get(text, self.state_to_id[self.UNK_TOKEN])

    def decode_ingredient(self, id):
        return self.id_to_ing.get(id, self.UNK_TOKEN)

    def decode_state(self, id):
        return self.id_to_state.get(id, self.UNK_TOKEN)

    @property
    def ing_vocab_size(self):
        return len(self.ing_to_id)

    @property
    def state_vocab_size(self):
        return len(self.state_to_id)


if __name__ == "__main__":
    try:
        tokenizer = RecipeTasteTokenizer(
            ingredients_path="config/ingredients.txt", states_path="config/states.txt"
        )
        print("Initialized Tokenizer")
        print(f"Total ingredients: {tokenizer.ing_vocab_size}")
        print(f"Total states: {tokenizer.state_vocab_size}")

        # エンコードの実験
        ing_id = tokenizer.encode_ingredient("onion")
        state_id = tokenizer.encode_state("glaze")
        print(f"onion's id: {ing_id} | glaze の背番号: {state_id}")

        # デコードの実験（逆引き）
        print(f"id: {ing_id}'s ingredient is: {tokenizer.decode_ingredient(ing_id)}")

    except Exception as e:
        print(f"error: {e}")
        print("Please check the file path of the text file")
