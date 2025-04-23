import json
import aiofiles
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import deque
import contextlib
import time

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
processed_files = []
with open('non_empty_users.json', 'r') as f:
    processed_files = json.load(f)


async def read_messages(filename, directory='/home/ubuntu/tweets/'):
    result = []
    filepath = os.path.join(directory, filename)
    async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.loads(await f.read())
            if not data:
                return []

            for user_id, tweet in data.items():

                if 'text' in tweet:
                    result.append(tweet['text'])

                if 'quote' in tweet and 'text' in tweet['quote']:
                    result.append(tweet['quote']['text'] + "\n" + tweet['text'])

                if 'retweet' in tweet and 'text' in tweet['retweet']:
                    result.append(tweet['retweet']['text'])

        except Exception as e:
            print(f"Ошибка при обработке {filename}: {e}")
    return result


def group_messages_by_token_limit(messages, token_limit=60000, model_name="deepseek-chat"):
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
    return groups


semaphore = asyncio.Semaphore(50)


async def extract_keywords(text: str) -> str:
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
    async with semaphore:
        print("start")
        start = time.time()
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0
                }
            )
            print("Status:", r.status_code)
            print("Text", r.text)
            print("Headers:", r.headers)
            print(time.time() - start)
            return r.json()["choices"][0]["message"]["content"]


async def process_file(file: Path) -> Tuple[str, List[str]]:
    if file.is_file():
        screen_name = str(file.stem)
        try:
            name = df.loc[screen_name, 'name']
            description = df.loc[screen_name, 'description']
            info = f"Name: {name}, ScreenName: {screen_name}, Description: {description}"
            messages = [info] + [str(v) for v in (await read_messages(screen_name + '.json'))]

            groups = group_messages_by_token_limit(messages)
            if not groups:
                return screen_name, []
            keywords = await extract_keywords('\n '.join(groups[0]))
            processed_files.append(screen_name)
            return screen_name, json.loads(keywords)
        except Exception as e:
            print(f"[ERROR] {screen_name}: {str(e)}")
            return screen_name, []


async def process_folder(path: str = "/home/ubuntu/tweets/") -> Dict[str, List[str]]:
    folder = Path(path)
    order_df = pd.read_csv('result.csv')
    order = order_df.iloc[:, 0].dropna().astype(str).tolist()

    existing_files = {f.stem: f for f in folder.glob("*.json")}

    files_ordered = [
        existing_files[screen_name]
        for screen_name in order
        if screen_name in existing_files and screen_name not in processed_files
    ]
    results = {}

    tasks = [process_file(file) for file in files_ordered]
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        screen_name, keywords = await future
        if keywords:
            results[screen_name] = keywords

    return results


async def main():
    res = await process_folder()
    with open("type_result3.json", "w") as f:
        json.dump(res, f)


asyncio.run(main())
