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

                Tweet: "[test_tweet]"
            """

ten_shot_prompt = """You are an AI assistant trained to recognize whether a tweet is describing a real disaster or not.
            Please analyze the tweet and determine if it's about a real disaster or not.
            Respond with:
                1 if the tweet is about a real disaster.
                0 if the tweet is not about a real disaster.

                Tweet: "[test_tweet]"
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
        prompt[0]['content'] = one_shot_prompt.replace("[tweet]", tweet_shots[0])
    elif shots == 10:
        assert len(tweet_shots) >= 10
        prompt[0]['content'] = ten_shot_prompt
        for i in shots:
            prompt[0]['content'] = prompt[0]['content'].replace(f"[tweet]{i}", tweet_shots[i])
    else:
        raise ValueError(f"Shots must be within 0, 1, 10, but was {shots}")
    prompt[0]['content'] = prompt[0]['content'].replace("[test_tweet]", test_tweet)
    return prompt


def main(model, shots):
    # load api key from .env file and init GPT API
    load_dotenv()
    client_ = init_gpt_api()

    # load data
    test_tweets, test_ids = load_disaster_test_dataset("./datasets/disaster/test.csv")
    # print(test_tweets[:5])
    # print("\n", test_ids[:5])

    # Loop through the test tweets
    predictions = []
    for tweet in tqdm(test_tweets, desc="Making predictions"):
        prompt = construct_prompt(shots, [], tweet)
        predicted = call_gpt_api(client_, model, prompt)
        predictions.extend(predicted)

    # write predictions to csv
    df = pd.DataFrame({"id": test_ids, "target": predictions})
    df.to_csv(f"./datasets/disaster/test_predictions_gpt4_{shots}_shot.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Select the gpt model to use.")
    parser.add_argument("--shots", type=int, default=0, choices=[0, 1, 10], help="Select the number of example shots that are proesented to the API.")
    args = parser.parse_args()

    main(model=args.model, shots=args.shots)
