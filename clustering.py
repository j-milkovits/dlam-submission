from torch import Tensor
import torch
from src.models.Cluster import ClusterModel
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

"""
Idea use the clusters of the tweets to classify them
just use simple k means

"""


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def batch_embed_tweets(model, tweets, batch_size=32):
    num_tweets = len(tweets)
    # Pre-allocate the numpy array for embeddings
    embedding_dim = 1024
    tweets_emb = np.empty((num_tweets, embedding_dim), dtype=np.float32)

    for i in tqdm(range(0, num_tweets, batch_size), desc="Embedding tweets"):
        batch = tweets[i:i + batch_size]
        with torch.no_grad():  # Disable gradient computation
            batch_emb = model.get_embeddings(batch)

        # Convert to numpy and store directly in pre-allocated array
        tweets_emb[i:i + len(batch)] = batch_emb.cpu().numpy()

        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return tweets_emb


def calculate_num_keywords(df):
    num_unique_keywords = df['keyword'].nunique()
    return num_unique_keywords * 4 + 20


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv("./datasets/disaster/train.csv", sep=",")
    num_clusters = calculate_num_keywords(df)

    model = ClusterModel(num_clusters=num_clusters)

    tweets, targets = df["text"].to_list(), df["target"].to_list()
    # tweets, targets = tweets[:512], targets[:512]
    tweets_emb = batch_embed_tweets(model, tweets)

    kmeans = model.train(tweets_emb, targets)
    purity_map = model.purity_score()
    print(purity_map)
    mean_purity = sum([p["mean"] for p in purity_map.values()]) / len(purity_map.values())
    print("mean purity", mean_purity)

    df = pd.read_csv("./datasets/disaster/test.csv", sep=",")
    tweets = df["text"].to_list()
    # tweets, targets = tweets[:512], targets[:512]

    tweets_emb = batch_embed_tweets(model, tweets)
    predictions = model.predict(tweets_emb)
    df["target"] = predictions
    df.to_csv("./datasets/disaster/clustering_test_predictions.csv", index=False, columns=["id", "target"])


def search_cluster_num():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv("./datasets/disaster/train.csv", sep=",")
    base_num_clusters = calculate_num_keywords(df)

    tweets, targets = df["text"].to_list(), df["target"].to_list()

    # Compute embeddings once
    initial_model = ClusterModel(num_clusters=base_num_clusters)
    tweets_emb = batch_embed_tweets(initial_model, tweets)

    # Test different numbers of clusters
    cluster_range = range(base_num_clusters // 3, base_num_clusters * 3, base_num_clusters // 6)
    results = []

    for num_clusters in tqdm(cluster_range, desc="Testing cluster numbers"):
        model = ClusterModel(num_clusters=num_clusters)
        _ = model.train(tweets_emb, targets)
        purity_map = model.purity_score()
        mean_purity = sum([p["mean"] for p in purity_map.values()]) / len(purity_map.values())
        results.append({"num_clusters": num_clusters, "mean_purity": mean_purity})
        print(f"Clusters: {num_clusters}, Mean Purity: {mean_purity}")

    # Find the best number of clusters
    best_result = max(results, key=lambda x: x["mean_purity"])
    print(f"Best number of clusters: {best_result['num_clusters']} with mean purity: {best_result['mean_purity']}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot([r["num_clusters"] for r in results], [r["mean_purity"] for r in results], marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Mean Purity")
    plt.title("Mean Purity vs Number of Clusters")
    plt.savefig("cluster_purity_plot.png")
    plt.close()


if __name__ == "__main__":
    # search_cluster_num()
    main()
