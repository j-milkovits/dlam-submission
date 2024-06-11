from typing import Callable

import matplotlib.pyplot as plt
import nltk
import torch
import torch.nn as nn
from nltk import word_tokenize
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.DisasterDataset import (DisasterDataset,
                                      load_disaster_train_dataset)
from src.models.SimpleMLP import SimpleMLP
from src.utils.device import get_device
from src.utils.embedding_functions import averaged_bag_of_words
from src.utils.embeddings import load_word2vec

nltk.download("punkt")


tweets, targets = load_disaster_train_dataset("./datasets/disaster/train.csv")


word2vec = load_word2vec("./embeddings/GoogleNews-vectors-negative300.bin")

emb_fn: Callable[[list[str]], torch.Tensor] = lambda token_list: averaged_bag_of_words(
    token_list, word2vec
)


learning_rate = 0.002
batch_size = 32
epochs = 100

train_dataset = DisasterDataset(tweets, targets, emb_fn, word_tokenize)

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)


device = get_device()

model = SimpleMLP().to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()


model.train()

train_loss = []
train_accuracy = []
for epoch in range(epochs):

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
        batch_acc = ((pred > 0.5) == targ).float().mean().item()
        train_accuracy_batch[idx] = batch_acc

    train_loss.append(train_loss_batch.mean().item())
    train_accuracy.append(train_accuracy_batch.mean().item())


plt.plot(train_loss)
plt.title("Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

plt.plot(train_accuracy)
plt.title("Train Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.show()
