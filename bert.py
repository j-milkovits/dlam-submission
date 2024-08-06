import argparse
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from safetensors.torch import save_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup

from src.data.DisasterDataset import load_disaster_train_dataset, load_disaster_test_dataset


class BertClassifier(nn.Module):
    def __init__(self, bert_model="bert-base-uncased", num_classes=1, hidden_size=128):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
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


def train_bert_model(model, train_dataloader, device, num_epochs=3):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    train_accuracies = []
    batch_train_losses = []
    batch_train_accuracies = []

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

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}")

    return batch_train_losses, batch_train_accuracies


def plot_metrics(batch_train_losses, batch_train_accuracies, batch_size, num_epochs):
    plt.figure(figsize=(15, 10))

    # Plot losses
    plt.subplot(2, 1, 1)
    plt.plot(batch_train_losses, label='Train Loss (Batch)')
    plt.title('Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')

    # Mark the end of each epoch
    for i in range(1, num_epochs + 1):
        plt.axvline(x=i * len(batch_train_losses) // num_epochs, color='r', linestyle='--',
                    label='Epoch End' if i == 1 else "")

    plt.legend()

    # Plot accuracies
    plt.subplot(2, 1, 2)
    plt.plot(batch_train_accuracies, label='Train Accuracy (Batch)')
    plt.title('Accuracy')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')

    # Mark the end of each epoch
    for i in range(1, num_epochs + 1):
        plt.axvline(x=i * len(batch_train_accuracies) // num_epochs, color='r', linestyle='--',
                    label='Epoch End' if i == 1 else "")

    plt.legend()

    plt.tight_layout()
    plt.savefig('./plots/bert_disaster_classifier.png')
    plt.close()


def main(base_model="bert-base-uncased", batch_size=32, num_epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tweets, targets = load_disaster_train_dataset("./datasets/disaster/train.csv")

    tokenizer = BertTokenizer.from_pretrained(base_model)
    train_dataset = DisasterDatasetBERT(tweets, targets, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = BertClassifier().to(device)

    batch_train_losses, batch_train_accuracies = train_bert_model(model, train_dataloader, device, num_epochs)
    meta_data = {"base_model": base_model, "num_epochs": str(num_epochs), "batch_size": str(batch_size)}
    save_model(model, './models/bert_disaster_classifier', meta_data)

    plot_metrics(batch_train_losses, batch_train_accuracies, batch_size=32, num_epochs=num_epochs)

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
    df.to_csv("./datasets/disaster/test_predictions_bert.csv", index=False)


if __name__ == "__main__":
    main()
