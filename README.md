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
| #  | Model                                     | Embedding | F1 Score | Note                                                    | Author |
|----|-------------------------------------------|-----------|----------|---------------------------------------------------------|--------|
| 1  | Clustering                                | e5        | 0.80263  | Improved from 0.79619 with "query: " prefix             | Marlon |
| 2  | SimpleMLP                                 | e5        | 0.78915  | Decreased with "query: " prefix to 0.71314              | Marlon |
| 3  | SimpleMLP                                 | word2vec  | 0.78605  |                                                         | Marlon |
| 4  | Ensemble of 1, 2, 3                       | -         | 0.81887  | With query prefix for clustering but not for MLP for e5 | Marlon |
| 5  | Simple BERT Classifier                    | BERT      | 0.84216  |                                                         | Marlon |
| 6  | Ensemble of 1, 2, 3, 5, 5                 | -         | 0.84278  | With query prefix for clustering but not for MLP for e5 | Marlon |
| 7  | BERT Large Classifier                     | BERT      | 0.83604  | 4 Epochs & 64 Batch size                                | Marlon |
| 8  | Ensemble of 1, 2, 3, 5, 5 + 2* BERT large | -         | 0.84186  | With query prefix for clustering but not for MLP for e5 | Marlon |
| 9  | RNN                                       |           |          |                                                         | Seska  |
| 10 | LSTM                                      |           |          |                                                         | Seska  |
| 11 | API query (GPT4o-mini) zero shot          | -         | 0.76616  | Prompt can be found in the script                       | Markus |
| 12 | SimpleMLP with encoder Roberta            | Roberta   | 0.84002  | Epochs: 2 / Batchsize: 64                               | Jonas  |
| 13 | KNN                                       | Roberta   | 0.73919  | k: 5                                                    | Jonas  |
| 14 | Bernice and co.                           |           |          |                                                         | Fred   |
| 15 | API query (GPT4o-mini) one shot           | -         | 0.78884  |                                                         | Markus |
| 16 | API query (GPT4o-mini) ten shot           | -         | 0.74961  |                                                         | Markus |
| 17 | API query (GPT4o) one shot                | -         | 0.76432  |                                                         | Markus |
| 18 | BERT Ensemble                             | BERT      | 0.83726  | voting - 5 different seeds                              | Marlon |
| 19 | BERT Ensemble                             | BERT      | 0.84033  | voting - 11 different seeds                             | Marlon |
| 20 | BERT Ensemble                             | BERT      | 0.84002  | value averaging - 5 different seeds                     | Marlon |
| 21 | BERT Ensemble                             | BERT      | 0.84278  | value averaging - 21 different seeds - dropout rate 0.4 | Marlon |
