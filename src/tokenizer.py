# src/data_loader/tokenizer.py


class RecipeTasteTokenizer:
    def __init__(self):
        # 特殊トークンの定義
        self.PAD_TOKEN = "[PAD]"  # For Blank
        self.MASK_TOKEN = "[MASK]"  # For Training
        self.UNK_TOKEN = "[UNK]"  # For Unknown

        # 1. 材料の辞書（Vocab）
        self.ing_to_id = {self.PAD_TOKEN: 0, self.MASK_TOKEN: 1, self.UNK_TOKEN: 2}
        self.id_to_ing = {}

        # 2. 状態の辞書（Vocab）
        self.state_to_id = {self.PAD_TOKEN: 0, self.MASK_TOKEN: 1, self.UNK_TOKEN: 2}
        self.id_to_state = {}

        # 初期設定として、最低限の単語を登録（あとからファイル読み込みに拡張可能）
        self._build_initial_vocab()

    def _build_initial_vocab(self):
        # 初期材料の登録
        initial_ings = [
            "white_wine",
            "white_wine_vinegar",
            "butter",
            "red_wine",
            "veal",
            "black_pepper",
        ]
        for ing in initial_ings:
            if ing not in self.ing_to_id:
                self.ing_to_id[ing] = len(self.ing_to_id)

        # 初期状態の登録
        initial_states = ["raw", "stock", "paste", "Fried", "reduce"]
        for state in initial_states:
            if state not in self.state_to_id:
                self.state_to_id[state] = len(self.state_to_id)

        # 逆引き辞書の作成
        self.id_to_ing = {v: k for k, v in self.ing_to_id.items()}
        self.id_to_state = {v: k for k, v in self.state_to_id.items()}

    def encode_ingredient(self, text):
        """材料テキストをIDに変換（未知語は[UNK]）"""
        return self.ing_to_id.get(text, self.ing_to_id[self.UNK_TOKEN])

    def encode_state(self, text):
        """状態テキストをIDに変換（未知語は[UNK]）"""
        return self.state_to_id.get(text, self.state_to_id[self.UNK_TOKEN])

    @property
    def ing_vocab_size(self):
        return len(self.ing_to_id)

    @property
    def state_vocab_size(self):
        return len(self.state_to_id)
