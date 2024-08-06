import numpy as np
import pandas as pd


def main(files):
    """
    Ensemble predictions from different models
    :param files:
    :return:
    """
    dfs = [pd.read_csv(file) for file in files]
    predictions = np.array([df["target"].values for df in dfs])
    predictions = np.array(predictions).mean(axis=0)
    predictions = (predictions > 0.5) * 1
    df = pd.DataFrame({"id": dfs[0]["id"], "target": predictions})
    df.to_csv("./datasets/disaster/ensemble_predictions.csv", index=False)


if __name__ == "__main__":
    files = [
        "./datasets/disaster/clustering_test_predictions.csv",
        "./datasets/disaster/test_predictions_e5.csv",
        "./datasets/disaster/test_predictions_word2vec.csv",
    ]
    main(files)
