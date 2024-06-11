from typing import Dict

import numpy as np
import torch


def averaged_bag_of_words(
    token_list: list[str],
    embedding: Dict[str, list[float]],
    embedding_length: int = 300,
) -> torch.Tensor:
    mean_list: list[list[float]] = []
    for token in token_list:
        if token in embedding:
            mean_list.append(embedding[token])
    # no embedding could be computed
    if len(mean_list) == 0:
        return torch.zeros((embedding_length))

    mean_tensor = torch.tensor(np.array(mean_list))
    return mean_tensor.mean(dim=0)
