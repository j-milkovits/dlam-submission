import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# Load the predictions
predictions_df = pd.read_csv('datasets/disaster/train_predictions_bert.csv')

# Load the actual data
actual_df = pd.read_csv('datasets/disaster/train.csv')

# Merge the dataframes on the 'id' column
merged_df = pd.merge(predictions_df, actual_df, on='id', how='inner')

# Find misclassified examples
misclassified_df = merged_df[merged_df['target_x'] != merged_df['target_y']]

# Select relevant columns
result_df = misclassified_df[['id', 'keyword', 'location', 'text', 'target_y']]
result_df = result_df.rename(columns={'target_y': 'actual_target'})

# Add a column for the predicted target
result_df['predicted_target'] = misclassified_df['target_x']

# Sort by id
result_df = result_df.sort_values('id')

# Save the misclassified examples to a new CSV file
result_df.to_csv('datasets/disaster/misclassified_tweets.csv', index=False)

print(f"Number of misclassified tweets: {len(result_df)}")
print("Misclassified tweets have been saved to 'datasets/disaster/misclassified_tweets.csv'")

# Analysis
# Calculate misclassification rate
total_tweets = len(merged_df)
misclassified_tweets = len(result_df)
misclassification_rate = misclassified_tweets / total_tweets
print(f"\nMisclassification rate: {misclassification_rate:.2%}")

# Analyze effect of keyword
keyword_misclassification = result_df['keyword'].value_counts()
keyword_all = merged_df['keyword'].value_counts()

print("\nTop 10 keywords in misclassified tweets:")
print(keyword_misclassification.head(10))
print("\nTop 10 keywords in all tweets:")
print(keyword_all.head(10))

# Calculate misclassification rate for each keyword
keyword_misclassification_rate = (keyword_misclassification / keyword_all).fillna(0)
print("\nTop 10 keywords with highest misclassification rate:")
print(keyword_misclassification_rate.sort_values(ascending=False).head(10))

# Analyze effect of location
location_misclassification = result_df['location'].value_counts()
location_all = merged_df['location'].value_counts()

print("\nTop 10 locations in misclassified tweets:")
print(location_misclassification.head(10))
print("\nTop 10 locations in all tweets:")
print(location_all.head(10))

# Calculate misclassification rate for each location
location_misclassification_rate = (location_misclassification / location_all).fillna(0)
print("\nTop 10 locations with highest misclassification rate:")
print(location_misclassification_rate.sort_values(ascending=False).head(10))

# Analyze tweet length
result_df['tweet_length'] = result_df['text'].str.len()
avg_misclassified_length = result_df['tweet_length'].mean()
avg_correct_length = merged_df[merged_df['target_x'] == merged_df['target_y']]['text'].str.len().mean()
print(f"\nAverage length of misclassified tweets: {avg_misclassified_length:.2f}")
print(f"Average length of correctly classified tweets: {avg_correct_length:.2f}")

# Visualize misclassification by tweet length
plt.figure(figsize=(10, 6))
plt.hist(result_df['tweet_length'], bins=20, alpha=0.5, label='Misclassified')
plt.hist(merged_df[merged_df['target_x'] == merged_df['target_y']]['text'].str.len(), bins=20, alpha=0.5,
         label='Correct')
plt.xlabel('Tweet Length')
plt.ylabel('Frequency')
plt.title('Distribution of Tweet Lengths')
plt.legend()
plt.savefig('plots/tweet_length_distribution.png')
plt.close()


# Analyze common words in misclassified tweets
def get_common_words(texts, n=10):
    words = [word for text in texts for word in text.lower().split()]
    return Counter(words).most_common(n)


misclassified_common_words = get_common_words(result_df['text'])
correct_common_words = get_common_words(merged_df[merged_df['target_x'] == merged_df['target_y']]['text'])

print("\nTop 10 common words in misclassified tweets:")
print(misclassified_common_words)
print("\nTop 10 common words in correctly classified tweets:")
print(correct_common_words)

# Analyze misclassification by prediction direction
false_positives = result_df[result_df['predicted_target'] > result_df['actual_target']]
false_negatives = result_df[result_df['predicted_target'] < result_df['actual_target']]
print(f"\nFalse positives: {len(false_positives)}")
print(f"False negatives: {len(false_negatives)}")


# Analyze effect of links
def contains_link(text):
    return 'http://' in text or 'https://' in text


merged_df['contains_link'] = merged_df['text'].apply(contains_link)
result_df['contains_link'] = result_df['text'].apply(contains_link)

link_misclassification_rate = len(result_df[result_df['contains_link']]) / len(result_df)
link_overall_rate = merged_df['contains_link'].mean()

print(f"\nProportion of misclassified tweets containing links: {link_misclassification_rate:.2%}")
print(f"Proportion of all tweets containing links: {link_overall_rate:.2%}")


# Analyze effect of hashtags
def count_hashtags(text):
    return len(re.findall(r'#\w+', text))


merged_df['hashtag_count'] = merged_df['text'].apply(count_hashtags)
result_df['hashtag_count'] = result_df['text'].apply(count_hashtags)

avg_hashtags_misclassified = result_df['hashtag_count'].mean()
avg_hashtags_overall = merged_df['hashtag_count'].mean()

print(f"\nAverage number of hashtags in misclassified tweets: {avg_hashtags_misclassified:.2f}")
print(f"Average number of hashtags in all tweets: {avg_hashtags_overall:.2f}")

# Visualize hashtag distribution
plt.figure(figsize=(10, 6))
plt.hist(result_df['hashtag_count'], bins=range(0, 10), alpha=0.5, label='Misclassified')
plt.hist(merged_df['hashtag_count'], bins=range(0, 10), alpha=0.5, label='All Tweets')
plt.xlabel('Number of Hashtags')
plt.ylabel('Frequency')
plt.title('Distribution of Hashtags in Tweets')
plt.legend()
plt.savefig('plots/hashtag_distribution.png')
plt.close()


# Analyze effect of words in all caps
def count_caps_words(text):
    return len([word for word in text.split() if word.isupper() and len(word) > 1])


merged_df['caps_word_count'] = merged_df['text'].apply(count_caps_words)
result_df['caps_word_count'] = result_df['text'].apply(count_caps_words)

avg_caps_misclassified = result_df['caps_word_count'].mean()
avg_caps_overall = merged_df['caps_word_count'].mean()

print(f"\nAverage number of words in all caps in misclassified tweets: {avg_caps_misclassified:.2f}")
print(f"Average number of words in all caps in all tweets: {avg_caps_overall:.2f}")

result_df['tweet_length'] = result_df['text'].str.len()

# Visualize caps words distribution
plt.figure(figsize=(10, 6))
plt.hist(result_df['caps_word_count'], bins=range(0, 10), alpha=0.5, label='Misclassified')
plt.hist(merged_df['caps_word_count'], bins=range(0, 10), alpha=0.5, label='All Tweets')
plt.xlabel('Number of Words in All Caps')
plt.ylabel('Frequency')
plt.title('Distribution of Words in All Caps in Tweets')
plt.legend()
plt.savefig('plots/caps_words_distribution.png')
plt.close()

# Analyze correlation between these features and misclassification
feature_correlations = result_df[['contains_link', 'hashtag_count', 'caps_word_count', 'tweet_length']].corr()
print("\nCorrelation between features in misclassified tweets:")
print(feature_correlations)

print("\nAnalysis complete.")
