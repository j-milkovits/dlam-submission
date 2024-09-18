import argparse
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from collections import Counter
import gensim
from src.data.DisasterDataset import load_disaster_train_dataset, load_disaster_test_dataset

def build_vocab(texts, max_vocab_size=5000):
    counter = Counter()
    for text in texts:
        tokens = text.split()
        counter.update(tokens)

    most_common = counter.most_common(max_vocab_size - 2) 
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for idx, (word, _) in enumerate(most_common, 2):
        word2idx[word] = idx
    return word2idx

word2vec_model_path = 'GoogleNews-vectors-negative300.bin'
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)

def load_pretrained_word2vec(word2idx, embedding_dim=300):
    """Load Word2Vec embeddings and create an embedding matrix for the vocabulary."""
    vocab_size = len(word2idx)
    embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim)) 

    for word, idx in word2idx.items():
        if word in word2vec:
            embedding_matrix[idx] = word2vec[word]
    
    return torch.tensor(embedding_matrix, dtype=torch.float32)

class RNNClassifier(nn.Module):
    def __init__(self, word2idx, embedding_dim=300, hidden_dim=256, num_layers=1, num_classes=1, bidirectional=True, seed=0):
        super(RNNClassifier, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        vocab_size = len(word2idx) 
        
        embedding_matrix = load_pretrained_word2vec(word2idx, embedding_dim)

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)  

        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, 
                          batch_first=True, bidirectional=bidirectional, dropout=0.3)

        rnn_output_size = hidden_dim * 2 if bidirectional else hidden_dim

        self.fc = nn.Sequential(
            nn.LayerNorm(rnn_output_size),  
            nn.Dropout(0.3),                
            nn.Linear(rnn_output_size, 128), 
            nn.ReLU(),                       
            nn.Linear(128, 64),              
            nn.ReLU(),                       
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)       
        )

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)

        rnn_out, hidden = self.rnn(embeddings)

        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1] 

        output = self.fc(hidden)
        return output



class DisasterDatasetRNN(Dataset):
    def __init__(self, texts, targets, word2idx, max_length=128):
        self.texts = texts
        self.targets = targets
        self.word2idx = word2idx
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]

        tokens = text.split() 
        token_ids = [self.word2idx.get(token, 1) for token in tokens] 

        if len(token_ids) < self.max_length:
            token_ids.extend([0] * (self.max_length - len(token_ids))) 
        else:
            token_ids = token_ids[:self.max_length]

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.float)
        }


def train_rnn_model(model, train_dataloader, val_dataloader, device, num_epochs=10):
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
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
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs.squeeze(dim=-1), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            predicted = (outputs.squeeze(dim=-1) > 0.5).float()
            epoch_train_total += targets.size(0)
            epoch_train_correct += (predicted == targets).sum().item()

            batch_train_losses.append(loss.item())
            batch_train_accuracies.append((predicted == targets).float().mean().item())

        epoch_train_loss /= len(train_dataloader)
        epoch_train_accuracy = epoch_train_correct / epoch_train_total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                targets = batch['target'].to(device)

                outputs = model(input_ids)
                loss = criterion(outputs.squeeze(dim=-1), targets)

                val_loss += loss.item()
                predicted = (outputs.squeeze(dim=-1) > 0.5).float()
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
    plt.savefig('./plots/rnn_disaster_classifier.png')
    plt.close()


def train_wrapper(base_model, batch_size, device, num_epochs, seed):
    tweets, targets = load_disaster_train_dataset("./datasets/disaster/train.csv")
    train_tweets, val_tweets, train_targets, val_targets = train_test_split(tweets, targets, test_size=0.2, random_state=seed)

    word2idx = build_vocab(train_tweets)
    vocab_size = len(word2idx)

    train_dataset = DisasterDatasetRNN(train_tweets, train_targets, word2idx)
    val_dataset = DisasterDatasetRNN(val_tweets, val_targets, word2idx)

    class_weights = compute_class_weight('balanced', classes=np.unique(train_targets), y=train_targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    model = RNNClassifier(word2idx=word2idx, seed=seed).to(device)

    batch_train_losses, batch_train_accuracies, val_losses, val_accuracies = train_rnn_model(
        model, train_dataloader, val_dataloader, device, num_epochs
    )

    return batch_train_accuracies, batch_train_losses, model, word2idx, val_accuracies, val_losses


def evaluate(batch_size, device, model, word2idx, save_path="./datasets/disaster/test_predictions_rnn6.csv"):
    test_tweets, test_ids = load_disaster_test_dataset("./datasets/disaster/test.csv")
    test_dataset = DisasterDatasetRNN(test_tweets, [0] * len(test_tweets), word2idx)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Making predictions"):
            input_ids = batch['input_ids'].to(device)
            outputs = model(input_ids)
            predicted = (outputs.squeeze() > 0.5).int().cpu().numpy().tolist()
            predictions.extend(predicted)
    df = pd.DataFrame({"id": test_ids, "target": predictions})
    df.to_csv(save_path, index=False)


def main(batch_size=16, num_epochs=10):
    seed = 7219
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    batch_acc, batch_loss, model, word2idx, val_acc, val_loss = train_wrapper(None, batch_size, device, num_epochs, seed)
    plot_metrics(batch_loss, batch_acc, val_loss, val_acc, batch_size, num_epochs)
    evaluate(batch_size, device, model, word2idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNN model for disaster tweet classification")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=8, help="Number of epochs to train")
    args = parser.parse_args()

    main(batch_size=args.batch_size, num_epochs=args.num_epochs)
