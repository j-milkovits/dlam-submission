


"""
Idea use the clusters of the tweets to classify them
just use simple k means

"""
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

#
#
# import torch.nn.functional as F
from torch import Tensor
# from transformers import AutoTokenizer, AutoModel
#
# def average_pool(last_hidden_states: Tensor,
#                  attention_mask: Tensor) -> Tensor:
#     last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
#     return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
#
#
# # Each input text should start with "query: " or "passage: ".
# # For tasks other than retrieval, you can simply use the "query: " prefix.
# input_texts = ["This is Bob, hello Bob.", "fnaiusojapo sakd ksad\þſæ asd ad"]
# tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
# model = AutoModel.from_pretrained('intfloat/e5-large-v2')
# model.to('cuda')
# # Tokenize the input texts
# batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
# batch_dict.to('cuda')
# outputs = model(**batch_dict)
# embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
#
# print(scores.tolist())


import torch
from src.data.DisasterDataset import (DisasterDataset,
                                      load_disaster_train_dataset)
from src.models.Cluster import ClusterModel
import pandas as pd
import numpy as np
from tqdm import tqdm


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def batch_embed_tweets(model, tweets, batch_size=32):
    num_tweets = len(tweets)
    # Pre-allocate the numpy array for embeddings
    embedding_dim = 1024
    tweets_emb = np.empty((num_tweets, embedding_dim), dtype=np.float32)

    for i in tqdm(range(0, num_tweets, batch_size)):
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
    return num_unique_keywords * 2 + 20


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv("./datasets/disaster/train.csv", sep=",")
    num_clusters = calculate_num_keywords(df)

    model = ClusterModel(num_clusters=50)

    tweets, targets = df["text"].to_list(), df["target"].to_list()
    # tweets, targets = tweets[:512], targets[:512]
    tweets_emb = batch_embed_tweets(model, tweets)

    kmeans = model.train(tweets_emb, targets)
    purity_map = model.purity_score()
    print(purity_map)
    print("mean purity", sum(purity_map.values()) / len(purity_map.values()))

    # kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(embeddings)















