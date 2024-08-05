from typing import Callable, Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class DisasterDataset(Dataset):
    def __init__(
        self,
        tweets: list[str],
        targets: list[int],
        emb_fn: Callable[[list[str]], torch.Tensor],
        tokenizer: Callable[[str], list[str]],
    ) -> None:
        self.data: list[Dict[str, torch.Tensor]] = []
        assert len(tweets) == len(
            targets
        ), "The arrays tweets and targets should have the same length."
        zipped_items = zip(tweets, targets)
        for tweet, target in tqdm(zipped_items, desc="Tokenizing and Embedding", total=len(tweets)):
            self.data.append(
                {
                    # tokenize and embed sentence
                    "tweet": emb_fn(tokenizer(tweet)),
                    "target": torch.tensor(target, dtype=torch.float),
                }
            )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def load_disaster_train_dataset(path: str) -> Tuple[
    list[str],
    list[int],
]:
    df = pd.read_csv(path, sep=",")
    return (
        list(df["text"].values),
        list(df["target"].values),
    )


def load_disaster_test_dataset(path: str) -> Tuple[
    list[str],
    list[int],
]:
    df = pd.read_csv(path, sep=",")
    return (
        list(df["text"].values),
        list(df["id"].values),
    )
