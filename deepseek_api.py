import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
KEY = os.getenv("DEEPSEEK_KEY")
client = OpenAI(api_key=KEY, base_url="https://api.deepseek.com")


def group_messages_by_token_limit(messages, token_limit=50000):
    groups = []
    current_group = []
    current_tokens = 0

    for message in messages:
        tokens = len(message)

        if current_tokens + tokens <= token_limit:
            current_group.append(message)
            current_tokens += tokens
        else:
            groups.append(current_group)
            current_group = [message]
            current_tokens = tokens

    if current_group:
        groups.append(current_group)

    return groups


def extract_keywords(text: str) -> str:
    prompt = (
        "Extract only the words and phrases related to cryptocurrencies "
        "(e.g., cryptocurrency names, exchanges, blockchain terms) and nothing more. "
        "Return them as a JSON list, with no additional explanations. "
        "If nothing is found, return an empty list. "
        "The result should be in the format: [\"word1\", ..., \"wordn\"] without '''json on the begining. "
        f"Text: {text}"
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content


def process_folder(path: str = "E:/users_data/tweets"):
    result = {}

    folder = Path(path)
    files = folder.glob("*.json")
    for i, file in tqdm(enumerate(files)):
        if file.is_file():
            content = file.read_text(encoding='utf-8')
            data = json.loads(content).values()

            groups = group_messages_by_token_limit(data)

            words = []
            try:
                for group in groups:
                    words.extend(json.loads(extract_keywords('. '.join(group))))
            except Exception as e:
                pass
            result[str(file.stem)] = words
    return result


res = process_folder()
json.dump(res, open("tmp.json", "w"))
