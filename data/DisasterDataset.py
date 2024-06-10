from typing import Callable, Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class DisasterDataset(Dataset):
    def __init__(
        self,
        tweets: list[str],
        targets: list[int],
        emb_fn: Callable[[list[str]], list[float]],
        tokenizer: Callable[[str], list[str]],
    ) -> None:
        self.data: list[Dict[str, torch.FloatTensor]] = []
        assert len(tweets) == len(
            targets
        ), "The arrays tweets and targets should have the same length."
        zipped_items = zip(tweets, targets)
        for tweet, target in zipped_items:
            self.data.append(
                {
                    # tokenize and embed sentence
                    "tweet": torch.tensor(
                        emb_fn(tokenizer(tweet)), dtype=torch.float32
                    ),
                    "target": torch.tensor(target, dtype=torch.float32),
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


def load_disaster_test_dataset(path: str) -> list[str]:
    df = pd.read_csv(path, sep=",")
    return list(df["text"].values)
