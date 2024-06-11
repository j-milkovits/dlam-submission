from typing import Dict

import gensim


def load_word2vec(path: str) -> Dict[str, list[float]]:
    return gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
