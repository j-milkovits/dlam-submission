import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from safetensors.torch import save_model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

from src.data.DisasterDataset import (load_disaster_test_dataset,
                                      load_disaster_train_dataset)


class RobertaEmbedder(nn.Module):
    def __init__(self, roberta_model="roberta-large", seed=8739):
        super(RobertaEmbedder, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.roberta = RobertaModel.from_pretrained(roberta_model)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return pooled_output


class DisasterDatasetRoberta(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length=128):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]

        # Add special tokens for better context understanding
        text = f"Tweet: {text}"

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "target": torch.tensor(target, dtype=torch.float),
        }


def extract_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            embeddings = model(input_ids, attention_mask)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(batch["target"].cpu().numpy())
    return np.concatenate(all_embeddings), np.array(all_labels)


def train_knn_classifier(train_embeddings, train_labels, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_embeddings, train_labels)
    return knn


def train_wrapper(base_model, batch_size, device, seed):
    tweets, targets = load_disaster_train_dataset("./datasets/disaster/train.csv")
    train_tweets, val_tweets, train_targets, val_targets = train_test_split(
        tweets, targets, test_size=0.2, random_state=seed
    )
    tokenizer = RobertaTokenizer.from_pretrained(base_model)
    train_dataset = DisasterDatasetRoberta(train_tweets, train_targets, tokenizer)
    val_dataset = DisasterDatasetRoberta(val_tweets, val_targets, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    model = RobertaEmbedder(roberta_model=base_model, seed=seed).to(device)

    # Extract embeddings
    train_embeddings, train_labels = extract_embeddings(model, train_dataloader, device)
    val_embeddings, val_labels = extract_embeddings(model, val_dataloader, device)

    # Train KNN classifier
    knn = train_knn_classifier(train_embeddings, train_labels)

    # Evaluate on validation set
    val_predictions = knn.predict(val_embeddings)
    val_accuracy = np.mean(val_predictions == val_labels)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    return knn, model, tokenizer


def evaluate(
    knn,
    model,
    tokenizer,
    batch_size,
    device,
    save_path="./datasets/disaster/test_predictions_knn.csv",
):
    test_tweets, test_ids = load_disaster_test_dataset("./datasets/disaster/test.csv")
    test_dataset = DisasterDatasetRoberta(
        test_tweets,
        [0] * len(test_tweets),
        tokenizer,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Extract test embeddings
    test_embeddings, _ = extract_embeddings(model, test_dataloader, device)

    # Predict using KNN
    test_predictions = knn.predict(test_embeddings).astype(int)

    # Save predictions
    df = pd.DataFrame({"id": test_ids, "target": test_predictions})
    df.to_csv(save_path, index=False)


def main(base_model="roberta-large", batch_size=32, seed=7219):
    knn, model, tokenizer = train_wrapper(base_model, batch_size, device, seed)
    evaluate(knn, model, tokenizer, batch_size, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Roberta model with KNN for disaster tweet classification"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="roberta-large",
        help="Base Roberta model to use",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for embedding extraction"
    )
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(args.base_model, args.batch_size)
