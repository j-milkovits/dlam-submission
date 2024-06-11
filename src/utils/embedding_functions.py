from typing import Dict

import numpy as np
import torch


def averaged_bag_of_words(
    token_list: list[str], embedding: Dict[str, list[float]]
) -> torch.Tensor:
    mean_list: list[list[float]] = []
    for token in token_list:
        if token in embedding:
            mean_list.append(embedding[token])
    mean_tensor = torch.tensor(np.array(mean_list))
    return mean_tensor.mean(dim=0)
