import json
import os
import threading
import asyncio
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

load_dotenv()
KEY = os.getenv("DEEPSEEK_KEY")
thread_local = threading.local()


async def create_client(max_workers: int = 255):
    limits = httpx.Limits(
        max_connections=max_workers,
        max_keepalive_connections=max_workers
    )

    return AsyncOpenAI(api_key=KEY, base_url="https://api.deepseek.com", http_client=httpx.AsyncClient(limits=limits))


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


async def extract_keywords(text: str, client) -> str:
    prompt = (
        "Extract only the words and phrases related to cryptocurrencies "
        "(e.g., cryptocurrency names, exchanges, blockchain terms) and nothing more. "
        "Return them as a JSON list, with no additional explanations. "
        "If nothing is found, return an empty list. "
        "The result should be in the format: [\"word1\", ..., \"wordn\"] without '''json on the begining. "
        f"Text: {text}"
    )

    response = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content


async def process_file(file, client):
    if file.is_file():
        content = file.read_text(encoding='utf-8')
        data = json.loads(content).values()

        groups = group_messages_by_token_limit(data)

        words = []
        try:
            for group in groups:
                words.extend(json.loads(await extract_keywords('. '.join(group), client)))
        except Exception as e:
            pass
        return str(file.stem), words


async def process_folder(path: str = "E:/users_data/tweets"):
    result = {}

    folder = Path(path)
    files = folder.glob("*.json")

    async with await create_client() as client:
        tasks = [
            process_file(file, client) for file in files
        ]

        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            filename, words = await future
            result[filename] = words
    return result


async def main():
    res = await process_folder()
    with open("tmp.json", "w") as f:
        json.dump(res, f)


asyncio.run(main())
