import argparse
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from safetensors.torch import save_model
from sklearn.metrics import confusion_matrix
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import wandb

from src.data.DisasterDataset import load_disaster_train_dataset, load_disaster_test_dataset


class BertClassifier(nn.Module):
    def __init__(self, bert_model="bert-base-uncased", num_classes=1, seed=8739, dropout=0.2):
        super(BertClassifier, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.bert = BertModel.from_pretrained(bert_model)
        self.layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.bert.config.hidden_size, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.layer(pooled_output)


class DisasterDatasetBERT(Dataset):
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
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'target': torch.tensor(target, dtype=torch.float)
        }


def train_bert_model(model, train_dataloader, val_dataloader, device, num_epochs=3):
    wandb.watch(model, log_freq=100)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        epoch_train_correct = 0
        epoch_train_total = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            epoch_train_total += targets.size(0)
            epoch_train_correct += (predicted == targets).sum().item()

            wandb.log({"batch_train_loss": loss.item(), "batch_train_accuracy": (predicted == targets).float().mean().item()})

        epoch_train_loss /= len(train_dataloader)
        epoch_train_accuracy = epoch_train_correct / epoch_train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['target'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), targets)

                val_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_loss /= len(val_dataloader)
        val_accuracy = val_correct / val_total

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "train_accuracy": epoch_train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return None


def train_wrapper(base_model, batch_size, device, num_epochs, seed, test_size=0.15, dropout=0.2):
    tweets, targets = load_disaster_train_dataset("./datasets/disaster/train.csv")
    train_tweets, val_tweets, train_targets, val_targets = train_test_split(tweets, targets, test_size=test_size,
                                                                            random_state=seed)
    tokenizer = BertTokenizer.from_pretrained(base_model)
    train_dataset = DisasterDatasetBERT(train_tweets, train_targets, tokenizer)
    val_dataset = DisasterDatasetBERT(val_tweets, val_targets, tokenizer)
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_targets), y=train_targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    # Create weighted sampler for imbalanced dataset
    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    model = BertClassifier(bert_model=base_model, seed=seed, dropout=dropout).to(device)
    train_bert_model(model, train_dataloader, val_dataloader, device, num_epochs)
    meta_data = {"base_model": base_model, "num_epochs": str(num_epochs), "batch_size": str(batch_size), "seed": str(seed)}
    save_model(model, './models/bert_disaster_classifier', meta_data)
    return model, tokenizer


def evaluate(batch_size, device, model, tokenizer, save_path="./datasets/disaster/test_predictions_bert.csv"):
    test_tweets, test_ids = load_disaster_test_dataset("./datasets/disaster/test.csv")
    test_dataset = DisasterDatasetBERT(test_tweets, [0] * len(test_tweets), tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Making predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            predicted = (outputs.squeeze() > 0.5).int().cpu().numpy().tolist()
            predictions.extend(predicted)
    df = pd.DataFrame({"id": test_ids, "target": predictions})
    df.to_csv(save_path, index=False)


def main(base_model="bert-base-uncased", batch_size=32, num_epochs=3, seed=7219, project_name="disaster-tweet-classification"):
    wandb.init(project=project_name, config={
        "base_model": base_model,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "seed": seed
    })

    model, tokenizer = train_wrapper(base_model, batch_size, device, num_epochs, seed)
    evaluate(batch_size, device, model, tokenizer)
    wandb.finish()


def train_ensemble(args, device, threshold=0.5, save_path="./datasets/disaster/bert_ensembles.csv", seed=4868):
    tokenizer = BertTokenizer.from_pretrained(args.base_model)

    test_tweets, test_ids = load_disaster_test_dataset("./datasets/disaster/test.csv")
    test_dataset = DisasterDatasetBERT(test_tweets, [0] * len(test_tweets), tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    all_logits = []

    for i in range(args.ensemble):
        wandb.init(project=args.project_name, name=f"ensemble_run_{i}", config={
            "base_model": args.base_model,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "dropout": args.dropout,
            "seed": seed + i
        })
        model, _ = train_wrapper(args.base_model, args.batch_size, device, args.num_epochs, seed + i,
                                 dropout=args.dropout)

        # Evaluate model
        model_logits = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc=f"Evaluating model"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask)
                logits = outputs.squeeze().cpu().numpy()
                model_logits.extend(logits)

        all_logits.append(model_logits)
        wandb.finish()

    all_logits = np.array(all_logits)
    averaged_logits = np.mean(all_logits, axis=0)
    predictions = (averaged_logits > threshold).astype(int)

    df = pd.DataFrame({"id": test_ids, "target": predictions})
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT model for disaster tweet classification")
    parser.add_argument("--base_model", type=str, default="bert-base-uncased", help="Base BERT model to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for the model")
    parser.add_argument("--ensemble", type=int, default=0, help="Number of models to train for ensemble")
    parser.add_argument("--project_name", type=str, default="disaster-tweet-classification", help="Name of the wandb project")
    args = parser.parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    if args.ensemble < 0:
        main(args.base_model, args.batch_size, args.num_epochs)
    else:
        train_ensemble(args, device, save_path=f"./datasets/disaster/bert_ensembles_size_{args.ensemble}.csv")
