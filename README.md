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
| Model                                                     | Embedding | F1 Score | Note                                             |
|-----------------------------------------------------------|-----------|----------|--------------------------------------------------|
| Clustering                                                | e5        | 0.80263  | Improved from 0.79619 with "query: " prefix      |
| SimpleMLP                                                 | e5        | 0.78915  | Decreased with "query: " prefix to 0.71314       |
| SimpleMLP                                                 | word2vec  | 0.78605  |                                                  |
| Ensemble of Clustering (e5) and SimpleMLP (e5 & word2vec) | -         | 0.81887  | With query prefix for clustering but not for MLP |
