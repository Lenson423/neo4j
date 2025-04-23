import json
import os
import threading
import asyncio
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm
import tiktoken
import pandas as pd

load_dotenv()
KEY = os.getenv("DEEPSEEK_KEY")

df = pd.read_csv("all_users.csv")
df.set_index('screen_name', inplace=True)

async def create_client(max_workers: int = 255):
    limits = httpx.Limits(
        max_connections=max_workers,
        max_keepalive_connections=max_workers
    )
    timeout = httpx.Timeout(60)
    return AsyncOpenAI(api_key=KEY, base_url="https://api.deepseek.com", http_client=httpx.AsyncClient(limits=limits),
                       timeout=timeout, max_retries=5)


def group_messages_by_token_limit(messages, token_limit=100000, model_name="deepseek-chat"):
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    groups = []
    current_group = []
    current_tokens = 0

    for message in messages:
        tokens = len(enc.encode(message, disallowed_special=()))

        if current_tokens + tokens <= token_limit:
            current_group.append(message)
            current_tokens += tokens
        else:
            groups.append(current_group)
            current_group = [message]
            current_tokens = tokens

    if current_group:
        groups.append(current_group)
    print(len(enc.encode('. '.join(groups[0]), disallowed_special=())))
    return groups



async def extract_keywords(text: str, client) -> str:
    prompt = (
        "Analyze the following text and identify which crypto subcategories the person may belong to." 
        "Main categories: trader, memecoiner, DeFi enthusiast, NFT collector, builder, investor, miner, analyst."
        "If you are absolutely sure the person belongs to another crypto-related category, indicate that as well."
        "Multiple categories may be returned. Return only the list of categories, without explanations."
        "If there is insufficient information to determine any crypto-related category, "
        "return the most likely general persona type based on the text. Possible general categories include: "
        "politician, blogger, journalist, news outlet, sports account, media organization, influencer, company, "
        "tech enthusiast, artist, musician, parody account, or other clearly identifiable types. "
        "Do not invent a crypto category if none is clearly present. "
        "The result should be in the format: [\"word1\", ..., \"wordn\"] without '''json on the begining."
        f"Text: {text}'"
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
        screen_name = str(file.stem)
        name = df.loc[screen_name, 'name']
        description = df.loc[screen_name, 'description']
        info = f"Name: {name}, ScreenName: {screen_name}, Description: {description}"
        messages = [info] + [str(v) for v in json.loads(content).values()]

        groups = group_messages_by_token_limit(messages)

        words = []
        try:
            words.extend(json.loads(await extract_keywords('. '.join(groups[0]), client)))
        except Exception as e:
            print(f"[ERROR] Failed to extract keywords for {file.stem}: {e}")
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
    with open("type_result.json", "w") as f:
        json.dump(res, f)


asyncio.run(main())
