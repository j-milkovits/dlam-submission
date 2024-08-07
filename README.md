# DLAM Submission
## Embeddings
### word2vec
- [Download](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g)
- Extract and put in /embeddings

### FastText
- [Download](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip)
- Extract and put in /embeddings

## Models
### Clustering

### SimpleMLP


## Results
| # | Model                                     | Embedding | F1 Score | Note                                                    |
|---|-------------------------------------------|-----------|----------|---------------------------------------------------------|
| 1 | Clustering                                | e5        | 0.80263  | Improved from 0.79619 with "query: " prefix             |
| 2 | SimpleMLP                                 | e5        | 0.78915  | Decreased with "query: " prefix to 0.71314              |
| 3 | SimpleMLP                                 | word2vec  | 0.78605  |                                                         |
| 4 | Ensemble of 1, 2, 3                       | -         | 0.81887  | With query prefix for clustering but not for MLP for e5 |
| 5 | Simple BERT Classifier                    | BERT      | 0.84216  |                                                         |
| 6 | Ensemble of 1, 2, 3, 5, 5                 | -         | 0.84278  | With query prefix for clustering but not for MLP for e5 |
| 7 | BERT Large Classifier                     | BERT      | 0.83604  | 4 Epochs & 64 Batch size                                |
| 8 | Ensemble of 1, 2, 3, 5, 5 + 2* BERT large | -         | 0.84186  | With query prefix for clustering but not for MLP for e5 |
