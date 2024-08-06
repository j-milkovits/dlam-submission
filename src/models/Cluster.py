import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans


class ClusterModel:
    def __init__(self, device=None, model_name="intfloat/e5-large-v2", find_clusters=False, num_clusters: int = 50):
        """

        :param device:
        :param model_name:
        :param find_clusters: whether to calculate the number of clusters during training time or use predefined number
        :param num_clusters:
        """
        self.kmeans = None
        self.cluster_label_map = {}

        if num_clusters < 1:
            raise ValueError("num_clusters must be greater than 0")
        self.num_clusters = num_clusters if not find_clusters else None

        device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(device)

    def predict(self, inputs, input_is_text=False):
        if input_is_text:
            embeddings = self.get_embeddings(inputs).cpu().detach().numpy()
        else:
            embeddings = inputs
        predictions = self.kmeans.predict(embeddings)
        res = [self.purity_map[p]["label"] for p in predictions.tolist()]
        return res

    def get_embeddings(self, input_texts):
        batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict.to(self.device)
        with torch.no_grad():
            outputs = self.model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return embeddings

    def train(self, inputs, targets, find_clusters=True, num_clusters=50, input_is_text=False):
        if input_is_text:
            embeddings = self.get_embeddings(inputs).cpu().detach().numpy()
        else:
            embeddings = inputs
        if not self.num_clusters:
            self.find_num_clusters(inputs)
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(embeddings)
        self.kmeans = kmeans
        self.label_clusters(inputs, targets)
        return kmeans

    def label_clusters(self, text_inputs, targets):
        self.cluster_label_map = {c: [] for c in range(self.num_clusters)}
        for t, c in zip(targets, self.kmeans.labels_.tolist()):
            self.cluster_label_map[c].append(t)

    def find_num_clusters(self, dataset):
        pass

    def purity_score(self):
        purity_map = {c: {0: 0, 1: 0, "mean": 0, "label": None} for c in range(self.num_clusters)}
        for c_id, v in self.cluster_label_map.items():
            sum_of_ones = sum(v)
            sum_of_zeros = len(v) - sum_of_ones
            purity = max(sum_of_ones, sum_of_zeros) / len(v)
            purity_map[c_id][0] = sum_of_zeros
            purity_map[c_id][1] = sum_of_ones
            purity_map[c_id]["label"] = 0 if sum_of_zeros >= sum_of_ones else 1
            purity_map[c_id]["mean"] = purity
        self.purity_map = purity_map
        return purity_map


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
