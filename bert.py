import argparse
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from safetensors.torch import save_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from src.data.DisasterDataset import load_disaster_train_dataset, load_disaster_test_dataset


class BertClassifier(nn.Module):
    def __init__(self, bert_model="bert-base-uncased", num_classes=1, seed=8739):
        super(BertClassifier, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.bert = BertModel.from_pretrained(bert_model)
        self.layer = nn.Sequential(
            nn.Dropout(0.2),
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
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # Added weight decay
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = nn.BCEWithLogitsLoss()

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    batch_train_losses, batch_train_accuracies = [], []

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            epoch_train_total += targets.size(0)
            epoch_train_correct += (predicted == targets).sum().item()

            batch_train_losses.append(loss.item())
            batch_train_accuracies.append((predicted == targets).float().mean().item())

        epoch_train_loss /= len(train_dataloader)
        epoch_train_accuracy = epoch_train_correct / epoch_train_total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

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

        val_loss /= len(val_dataloader)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return batch_train_losses, batch_train_accuracies, val_losses, val_accuracies


def plot_metrics(batch_train_losses, batch_train_accuracies, val_losses, val_accuracies, batch_size, num_epochs):
    plt.figure(figsize=(15, 15))

    # Plot losses
    plt.subplot(3, 1, 1)
    plt.plot(batch_train_losses, label='Train Loss (Batch)')
    plt.title('Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')

    # Mark the end of each epoch
    for i in range(1, num_epochs + 1):
        plt.axvline(x=i * len(batch_train_losses) // num_epochs, color='r', linestyle='--',
                    label='Epoch End' if i == 1 else "")

    plt.legend()

    # Plot batch accuracies
    plt.subplot(3, 1, 2)
    plt.plot(batch_train_accuracies, label='Train Accuracy (Batch)')
    plt.title('Batch Accuracy')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')

    # Mark the end of each epoch
    for i in range(1, num_epochs + 1):
        plt.axvline(x=i * len(batch_train_accuracies) // num_epochs, color='r', linestyle='--',
                    label='Epoch End' if i == 1 else "")

    plt.legend()

    # Plot epoch accuracies
    plt.subplot(3, 1, 3)
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, [batch_train_accuracies[i * len(batch_train_accuracies) // num_epochs - 1] for i in epochs], 'bo-',
             label='Train')
    plt.plot(epochs, val_accuracies, 'ro-', label='Validation')
    plt.title('Epoch Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('./plots/bert_disaster_classifier.png')
    plt.close()


def train_wrapper(base_model, batch_size, device, num_epochs, seed):
    tweets, targets = load_disaster_train_dataset("./datasets/disaster/train.csv")
    train_tweets, val_tweets, train_targets, val_targets = train_test_split(tweets, targets, test_size=0.2,
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
    model = (BertClassifier(bert_model=base_model, seed=seed).to(device))
    batch_train_losses, batch_train_accuracies, val_losses, val_accuracies = train_bert_model(
        model, train_dataloader, val_dataloader, device, num_epochs
    )
    meta_data = {"base_model": base_model, "num_epochs": str(num_epochs), "batch_size": str(batch_size), "seed": str(seed)}
    save_model(model, './models/bert_disaster_classifier', meta_data)
    return batch_train_accuracies, batch_train_losses, model, tokenizer, val_accuracies, val_losses


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


def main(base_model="bert-base-uncased", batch_size=32, num_epochs=3, seed=7219):
    batch_acc, batch_loss, model, tokenizer, val_acc, val_loss = train_wrapper(base_model, batch_size, device, num_epochs, seed)
    plot_metrics(batch_loss, batch_acc, val_loss, val_acc, batch_size, num_epochs)
    evaluate(batch_size, device, model, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT model for disaster tweet classification")
    parser.add_argument("--base_model", type=str, default="bert-base-uncased", help="Base BERT model to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--ensemble", type=int, default=0, help="Number of models to train for ensemble")
    args = parser.parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    if args.ensemble < 0:
        main(args.base_model, args.batch_size, args.num_epochs)
    else:
        for i in range(args.ensemble):
            batch_acc, batch_loss, model, tokenizer, val_acc, val_loss = train_wrapper(args.base_model, args.batch_size, device, args.num_epochs, 4868 + i)
            evaluate(args.batch_size, device, model, tokenizer, save_path=f"./datasets/disaster/test_predictions_bert_{i}.csv")
