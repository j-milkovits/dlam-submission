import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_plot(plt, filename):
    ensure_dir("./plots")
    plt.savefig(f"./plots/{filename}", bbox_inches='tight')
    plt.close()


def load_and_explore_dataset(file_path, dataset_name):
    df = pd.read_csv(file_path, sep=",")
    df['keyword'] = df['keyword'].fillna('None').apply(lambda x: x.replace('%20', ' ') if x != 'None' else x)
    print(f"\n{dataset_name} dataset shape:", df.shape)
    print(f"\n{dataset_name} dataset columns:", df.columns)
    print(f"\n{dataset_name} dataset info:")
    df.info()
    print(f"\n{dataset_name} dataset sample:")
    print(df.head())
    return df


def analyze_target_distribution(df):
    target_counts = df['target'].value_counts()
    print("\nTarget distribution:")
    print(target_counts)

    plt.figure(figsize=(8, 6))
    target_counts.plot(kind='bar')
    plt.title("Distribution of Target Labels (Train)")
    plt.xlabel("Target")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    save_plot(plt, "target_distribution.png")


def analyze_feature(df, feature_name, dataset_type, top_n=20):
    none_count = (df[feature_name] == 'None').sum()
    nan_count = df[feature_name].isna().sum()
    total_missing = none_count + nan_count

    feature_counts = df[feature_name].replace('None', np.nan).value_counts(dropna=False)

    print(f"\nTop {top_n} {feature_name}s ({dataset_type}):")
    print(feature_counts.head(top_n))

    print(f"\nNumber of unique {feature_name}s ({dataset_type}):", len(feature_counts) - 1)  # Subtract 1 to exclude NaN
    print(f"Number of tweets without {feature_name} ({dataset_type}):", total_missing)
    print(f"  - 'None' values: {none_count}")
    print(f"  - NaN values: {nan_count}")

    plt.figure(figsize=(14, 8))
    feature_counts.head(top_n).plot(kind='bar')
    plt.title(f"Top {top_n} {feature_name.capitalize()}s ({dataset_type})")
    plt.xlabel(feature_name.capitalize())
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha='right')
    save_plot(plt, f"{feature_name}_distribution_{dataset_type}.png")


def compare_text_length_distribution(train_df, test_df):
    train_df['text_length'] = train_df['text'].str.len()
    test_df['text_length'] = test_df['text'].str.len()

    plt.figure(figsize=(10, 6))
    plt.hist(train_df['text_length'], bins=50, alpha=0.5, label='Train')
    plt.hist(test_df['text_length'], bins=50, alpha=0.5, label='Test')
    plt.title("Distribution of Text Length (Train vs Test)")
    plt.xlabel("Text Length")
    plt.ylabel("Count")
    plt.legend(loc='upper left')

    train_mean, train_std = train_df['text_length'].mean(), train_df['text_length'].std()
    test_mean, test_std = test_df['text_length'].mean(), test_df['text_length'].std()

    stats_text = f"Train: Mean={train_mean:.2f}, Std={train_std:.2f}\nTest: Mean={test_mean:.2f}, Std={test_std:.2f}"
    plt.text(0.5, 0.97, stats_text, transform=plt.gca().transAxes, verticalalignment='top',
             horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))

    save_plot(plt, "text_length_distribution.png")


def analyze_keyword_overlap(train_df, test_df):
    train_keywords = set(train_df['keyword'].dropna().unique())
    test_keywords = set(test_df['keyword'].dropna().unique())

    overlap = train_keywords.intersection(test_keywords)

    print("\nKeyword Analysis:")
    print(f"Number of unique keywords in train set: {len(train_keywords)}")
    print(f"Number of unique keywords in test set: {len(test_keywords)}")
    print(f"Number of overlapping keywords: {len(overlap)}")
    print(f"Percentage of train keywords in test: {len(overlap) / len(train_keywords) * 100:.2f}%")
    print(f"Percentage of test keywords in train: {len(overlap) / len(test_keywords) * 100:.2f}%")

    print("\nKeywords in test but not in train:")
    print(test_keywords - train_keywords)

    print("\nKeywords in train but not in test:")
    print(train_keywords - test_keywords)


def main():
    train_df = load_and_explore_dataset("./datasets/disaster/train.csv", "Training")
    test_df = load_and_explore_dataset("./datasets/disaster/test.csv", "Test")

    analyze_target_distribution(train_df)

    analyze_feature(train_df, "keyword", "Train")
    analyze_feature(test_df, "keyword", "Test")

    analyze_feature(train_df, "location", "Train")
    analyze_feature(test_df, "location", "Test")

    compare_text_length_distribution(train_df, test_df)

    analyze_keyword_overlap(train_df, test_df)


if __name__ == "__main__":
    main()
