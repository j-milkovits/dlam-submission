import argparse
from typing import Callable

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from torch import Tensor
import torch
import torch.nn as nn
from nltk import word_tokenize
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.data.DisasterDataset import (DisasterDataset, load_disaster_train_dataset, load_disaster_test_dataset)
from src.models.SimpleMLP import SimpleMLP
from src.utils.embedding_functions import averaged_bag_of_words
from src.utils.embeddings import load_word2vec


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def main(embedder: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if embedder == "word2vec":
        nltk.download("punkt")

        word2vec = load_word2vec("./embeddings/GoogleNews-vectors-negative300.bin")

        emb_fn: Callable[[list[str]], torch.Tensor] = lambda token_list: averaged_bag_of_words(
            token_list, word2vec
        )
        tokenizer = word_tokenize
        input_size = 300
    elif embedder == "e5":
        model_name = "intfloat/e5-large-v2"
        e5_model = AutoModel.from_pretrained(model_name)
        tokenizer2 = AutoTokenizer.from_pretrained(model_name)
        tokenizer = lambda x: tokenizer2(x, max_length=512, padding=True, truncation=True, return_tensors='pt')
        e5_model.to(device)
        emb_fn: Callable[[list[str]], torch.Tensor] = lambda token_list: get_embeddings(token_list)
        input_size = 1024

        def get_embeddings(input_texts):
            input_texts.to(device)
            with torch.no_grad():
                outputs = e5_model(**input_texts)
            embeddings = average_pool(outputs.last_hidden_state, input_texts['attention_mask'])
            return embeddings

    else:
        raise ValueError("Invalid embedder")

    learning_rate = 0.002
    batch_size = 32
    epochs = 200

    tweets, targets = load_disaster_train_dataset("./datasets/disaster/train.csv")
    # tweets, targets = tweets[:100], targets[:100]
    train_dataset = DisasterDataset(tweets, targets, emb_fn, tokenizer)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    model = SimpleMLP(input_size).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    model.train()

    train_loss = []
    train_accuracy = []
    for _ in tqdm(range(epochs), desc="Training Epochs"):

        train_loss_batch = torch.zeros((len(train_dataloader)))
        train_accuracy_batch = torch.zeros((len(train_dataloader)))

        for idx, batch in enumerate(train_dataloader):
            # prediction & loss
            pred = model(batch["tweet"].to(device))
            targ = batch["target"].to(device)
            loss = criterion(pred.squeeze(), targ)

            train_loss_batch[idx] = loss.item()

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy
            pred = ((pred > 0.5) == targ).float().mean().item()
            train_accuracy_batch[idx] = pred

        train_loss.append(train_loss_batch.mean().item())
        train_accuracy.append(train_accuracy_batch.mean().item())

    plot_loss(train_accuracy, train_loss, embedder)

    tweets, ids = load_disaster_test_dataset("./datasets/disaster/test.csv")
    test_dataset = DisasterDataset(tweets, [0] * len(tweets), emb_fn, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model.eval()

    predictions = []
    for idx, batch in enumerate(test_dataloader):
        pred = model(batch["tweet"].to(device))
        pred = (pred > 0.5).int().cpu().numpy().squeeze().tolist()
        predictions.extend(pred)

    predictions = [int(pred) for pred in predictions]
    df = pd.DataFrame({"id": ids, "target": predictions})
    df.to_csv(f"./datasets/disaster/test_predictions_{embedder}.csv", index=False)


def plot_loss(train_accuracy, train_loss, embedder):
    plt.plot(train_loss)
    plt.title(f"Train Loss: Simple MLP {embedder} embeddings")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(f"./plots/simple_mlp_loss{embedder}.png")
    plt.close()

    plt.plot(train_accuracy)
    plt.title(f"Train Accuracy: Simple MLP with {embedder} embeddings")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.savefig(f"./plots/simple_mlp_accuracy_{embedder}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", type=str, default="word2vec", help="Select the embedder to use.")
    args = parser.parse_args()

    main(embedder=args.embedder)
