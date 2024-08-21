from openai import OpenAI
from dotenv import load_dotenv
import os
import argparse


def call_gpt_api(client, model, prompt):
    completion = client.chat.completions.create(
        model=model,
        messages=prompt,
    )
    return completion.choices[0].message.content


def init_gpt_api():
    return OpenAI(api_key=os.getenv('api_key'))


def main(model):
    # load api key from .env file and init GPT API
    load_dotenv()
    client_ = init_gpt_api()

    prompt = [{"role": "user", "content": "What is the capital of Germany?"}]

    # example call
    print("answer\n", call_gpt_api(client_, model, prompt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Select the gpt model to use.")
    args = parser.parse_args()

    main(model=args.model)
