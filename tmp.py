import asyncio
import json
from pathlib import Path

from tqdm import tqdm


async def process_file(file):
    if file.is_file():
        content = file.read_text(encoding='utf-8')
        data = json.loads(content)
        return str(file.stem), data is None


async def process_folder(path: str = "/home/ubuntu/tweets/"):
    result = []

    folder = Path(path)
    files = folder.glob("*.json")

    tasks = [
        process_file(file) for file in files
    ]

    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        filename, flag = await future
        if flag:
            result.append(filename)
    return result


async def main():
    my_list = await process_folder()
    with open('output.txt', 'w') as file:
        for item in my_list:
            file.write(f"{item}\n")


asyncio.run(main())
