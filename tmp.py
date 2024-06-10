from typing import Callable, Dict

import numpy as np
import torch
from nltk import tokenize

from data.DisasterDataset import DisasterDataset, load_disaster_train_dataset
from models.SimpleMLP import SimpleMLP

tweets, targets = load_disaster_train_dataset("./dataset/disaster/train.csv")

# utils

def embed_token_list(tokens: list[str], emb: Dict[str, list[float]]):
    vectors: list[list[float]] = []
    for token in tokens:
        if token in emb:
            vectors.append(emb[token])
        else:
            vectors.append(list(np.zeros((300,))))
    return list(np.mean(vectors, axis=0))

train_dataset = DisasterDataset(tweets, targets, , tokenize)
