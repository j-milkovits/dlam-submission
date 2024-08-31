from openai import OpenAI
from dotenv import load_dotenv
import os
import argparse
import pandas as pd
from tqdm import tqdm

from src.data.DisasterDataset import load_disaster_train_dataset, load_disaster_test_dataset


zero_shot_prompt = """You are an AI assistant trained to recognize whether a tweet is describing a real disaster or not.
            Please analyze the tweet and determine if it's about a real disaster or not.
            Respond with:
                1 if the tweet is about a real disaster.
                0 if the tweet is not about a real disaster.

                Tweet: "[test_tweet]"
            """

one_shot_prompt = """You are an AI assistant trained to recognize whether a tweet is describing a real disaster or not.
            Please analyze the tweet and determine if it's about a real disaster or not.
            Respond with:
                1 if the tweet is about a real disaster.
                0 if the tweet is not about a real disaster.
                
                Here is an example:
                    Example Tweet: "[example_tweet]"
                    Desaster Classification: [desaster]

                Tweet for prediction: "[test_tweet]"
            """

ten_shot_prompt = """You are an AI assistant trained to recognize whether a tweet is describing a real disaster or not.
            Please analyze the tweet and determine if it's about a real disaster or not.
            Respond with:
                1 if the tweet is about a real disaster.
                0 if the tweet is not about a real disaster.

                Here are some examples:
                    Example Tweet 1: "[example_tweet0]" (Desaster Classification: [desaster0])
                    Example Tweet 2: "[example_tweet1]" (Desaster Classification: [desaster1])
                    Example Tweet 3: "[example_tweet2]" (Desaster Classification: [desaster2])
                    Example Tweet 4: "[example_tweet3]" (Desaster Classification: [desaster3])
                    Example Tweet 5: "[example_tweet4]" (Desaster Classification: [desaster4])
                    Example Tweet 6: "[example_tweet5]" (Desaster Classification: [desaster5])
                    Example Tweet 7: "[example_tweet6]" (Desaster Classification: [desaster6])
                    Example Tweet 8: "[example_tweet7]" (Desaster Classification: [desaster7])
                    Example Tweet 9: "[example_tweet8]" (Desaster Classification: [desaster8])
                    Example Tweet 10: "[example_tweet9]" (Desaster Classification: [desaster9])

                Tweet for prediction: "[test_tweet]"
            """

prompt_template = [{"role": "user", "content": "dummy"}]


def init_gpt_api():
    return OpenAI(api_key=os.getenv('api_key'))


def call_gpt_api(client, model, prompt):
    completion = client.chat.completions.create(
        model=model,
        messages=prompt,
    )
    return completion.choices[0].message.content


def construct_prompt(shots, tweet_shots, test_tweet):
    prompt = prompt_template
    if shots == 0:
        prompt[0]['content'] = zero_shot_prompt
    elif shots == 1:
        assert len(tweet_shots) >= 1
        prompt[0]['content'] = one_shot_prompt
        prompt[0]['content'] = prompt[0]['content'].replace("[example_tweet]", tweet_shots[0][0])
        prompt[0]['content'] = prompt[0]['content'].replace("[desaster]", str(tweet_shots[0][1]))
    elif shots == 10:
        assert len(tweet_shots) >= 10
        prompt[0]['content'] = ten_shot_prompt
        for i in range(shots):
            prompt[0]['content'] = prompt[0]['content'].replace(f"[example_tweet{i}]", tweet_shots[i][0])
            prompt[0]['content'] = prompt[0]['content'].replace(f"[desaster{i}]", str(tweet_shots[i][1]))
    else:
        raise ValueError(f"Shots must be within 0, 1, 10, but was {shots}")
    prompt[0]['content'] = prompt[0]['content'].replace("[test_tweet]", test_tweet)
    return prompt


def clean_response(response):
    if "0" in response:
        cleaned = "0"
    elif "1" in response:
        cleaned = "1"
    else:
        raise RuntimeError(f"Couldn't find prediction in API response: {response}")
    return cleaned


def main(model, shots):
    # load api key from .env file and init GPT API
    load_dotenv()
    client_ = init_gpt_api()

    # load data
    test_tweets, test_ids = load_disaster_test_dataset("./datasets/disaster/test.csv")
    train_tweets, targets = load_disaster_train_dataset("./datasets/disaster/train.csv")
    # print(test_tweets[:5])
    # print("\n", test_ids[:5])

    # Define the tweets from the train dataset that are provided to the prompt
    tweet_shots = []
    use_tweets_idx = [0, 1, 2, 3, 4, 20, 21, 22, 23, 24]
    for i in range(shots):
        tweet_shots.append((train_tweets[use_tweets_idx[i]], targets[use_tweets_idx[i]]))
    #print(tweet_shots)

    # Loop through the test tweets
    predictions = []
    for tweet in tqdm(test_tweets, desc="Making predictions"):
        prompt = construct_prompt(shots, tweet_shots, tweet)
        #print(prompt)
        predicted = call_gpt_api(client_, model, prompt)
        predicted = clean_response(predicted)
        predictions.extend(predicted)

    #print(predictions)
    # write predictions to csv
    df = pd.DataFrame({"id": test_ids, "target": predictions})
    df.to_csv(f"./datasets/disaster/test_predictions_gpt4_{shots}_shot.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Select the gpt model to use.")
    parser.add_argument("--shots", type=int, default=10, choices=[0, 1, 10], help="Select the number of example shots that are proesented to the API.")
    args = parser.parse_args()

    main(model=args.model, shots=args.shots)
