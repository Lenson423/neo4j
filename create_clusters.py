import asyncio
import json
import os
from collections import defaultdict

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from neo4j_dao import Neo4jDAO

load_dotenv()
KEY = os.getenv("DEEPSEEK_KEY")


async def create_client(max_workers: int = 40):
    limits = httpx.Limits(
        max_connections=max_workers,
        max_keepalive_connections=max_workers
    )

    return AsyncOpenAI(api_key=KEY, base_url="https://api.deepseek.com",
                       http_client=httpx.AsyncClient(limits=limits, http2=True))


def get_comments(path: str, basic: str = "/home/ubuntu/tweets/"):
    with open(basic + path + '.json', "r") as file:
        data = json.load(file)
        if data is not None:
            return data.values()
        else:
            return [""]


def get_messages_by_token_limit(messages, token_limit=70000):
    current_group = []
    current_tokens = 0

    for message in messages:
        tokens = len(message)

        if current_tokens + tokens <= token_limit:
            current_group.append(message)
            current_tokens += tokens
        else:
            break
    return current_group


async def is_follower_in_user_network(user_info, follower_info, follower_comments, client):
    prompt = f"""
            Determine whether the follower belongs to the thematic network of the main user.
            Return **only** `True` or `False`.
            If there is insufficient information to determine return `False`.
            
            User information:
            Name: {user_info['name']}
            Screen name: {user_info['screen_name']}
            Description: {user_info['description']}
            Tags: {", ".join(user_info['tags'])}
            
            Follower information:
            Name: {follower_info['name']}
            Screen name: {follower_info['screen_name']}
            Description: {follower_info['description']}
            Messages:
            {"".join(f"- {m}\n" for m in follower_comments)}
            """
    response = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return follower_info["screen_name"], response.choices[0].message.content == "True"


def get_all_centers(path: str = "type_result.json"):
    with open(path, "r") as file:
        data = json.load(file)
        return data


async def main():
    target_types = {
        "trader", "memecoiner", "DeFi enthusiast",
        "NFT collector", "builder", "investor",
        "miner", "analyst"
    }

    data = get_all_centers()

    matching_users = [
        (user, tags) for user, tags in data.items()
        if any(tag in target_types for tag in tags)
    ]

    result = defaultdict(list)

    neo = Neo4jDAO()
    async with await create_client() as client:
        for user, tags in matching_users:
            user_info = await neo.get_user_data(user)
            user_info["tags"] = tags

            df = await neo.get_all_followers(user)
            followers_data = df.to_dict(orient='records')

            tasks = [
                is_follower_in_user_network(user_info, follower_info,
                                            get_messages_by_token_limit(get_comments(follower_info["screen_name"])),
                                            client) for follower_info in followers_data
            ]

            for future in tqdm.as_completed(tasks):
                follower_name, ok = await future
                if ok:
                    result[user].append(follower_name)
            with open('res/' + user + '.json', 'w') as file:
                json.dump(result[user], file)
    await neo.close()


asyncio.run(main())
