import json
import os
import torch
from torch.utils.data import Dataset
from src.data_loader.tokenizer import RecipeTasteTokenizer


class RecipeTasteDataset(Dataset):
    def __init__(
        self,
        recipe_path="data/raw/recipes.json",
        config_path="config/defalt.yaml",
        max_len=8,
    ):
        # Create the instance of Tokenizer's RecipeTasteTokenizer
        self.tokenizer = RecipeTasteTokenizer
        self.max_len = 8

        if not os.path.exists(recipe_path):
            raise FileNotFoundError("File not found")

        with open(recipe_path, "r", encoding="utf-8") as f:
            self.recipes = json.load(f)

    def __len__(self):
        return len(self.recipes)

    def __getitem__(self, idx):
        recipe = self.recipes
        ingredients = recipe["ingredients"]

        ing_ids = []
        state_ids = []

        for item in ingredients:
            ing_ids.append(self.tokenizer.encode_ingredient(item["name"]))
            state_ids.append(self.tokenizer.encode_state(item["state"]))

            while len(ing_ids) < self.max_len:
                ing_ids.append(
                    self.tokenizer.tokenizer.ing_to_id["PAD"]
                    if hasattr(self.tokenizer, "tokenizer")
                    else 0
                )
                state_ids.append(
                    self.tokenizer.tokenizer.state_ids["PAD"]
                    if hasattr(self.tokenizer, "tokenizer")
                    else 0
                )
