import asyncio
import json
import os
import typing
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
import pandas as pd

from csv_processor import load_users
from userInfo import User

load_dotenv()
URI = os.getenv("URI")
AUTH = (os.getenv("AUTH_NAME"), os.getenv("AUTH_PASSWORD"))


class Neo4jDAO:  # ToDO
    driver: AsyncDriver

    def __init__(self):
        self.driver = AsyncGraphDatabase.driver(URI, auth=AUTH, max_connection_pool_size=20)

    async def close(self):
        await self.driver.close()

    async def get_session(self) -> AsyncSession:
        return self.driver.session(database="neo4j")

    async def add_empty_user(self, user_id: int, overwrite: bool = False, generation: int = 0):
        """
                Добавляет пользователя в базу данных.

                Если пользователь с указанным user_id уже существует:
                    - При overwrite=True помечает пользователч как обработанного и назначает ему новое поколение.
                    - При overwrite=False оставляет существующую запись без изменений.

                В противном случае создаёт нового пользователя с указанными полями.

                Args:
                    user_id (int): ID пользователя.
                    overwrite (bool, optional): Флаг, определяющий, нужно ли перезаписывать существующую запись.
                                                По умолчанию False.
                    generation (int, optional): Числовой параметр, указывающий поколение пользователя.
                                                По умолчанию 0.

                Returns:
                    None
                """
        async with (await self.get_session()) as session:
            if overwrite:
                await session.run("""
                                            MERGE (u:User {id: $id})
                                            SET u.needToProcess = $flag, u.generation = $generation
                                            """,
                                  {"id": user_id, "flag": True, "generation": generation})
            else:
                await session.run("""
                                            MERGE (u:User {id: $id})
                                            ON CREATE SET u.needToProcess = $flag, u.generation = $generation
                                            """,
                                  {"id": user_id, "flag": True, "generation": generation})

    async def add_user(self, user_id: int, subscriptions: np.array, create_followings: bool = True,
                       generation: int = 0, child_generation: int | None = None) -> None:
        """
                Добавляет пользователя в базу данных и устанавливает связи на основе отношения подписки.

                Метод выполняет следующие действия:
                1. Создаёт или обновляет пользователя с указанным user_id, и делаает его обработанным +
                   задааёт его поколение (generation).
                2. Для каждого идентификатора в subscriptions:
                   - Если create_followings=True, создаёт узел подписки (если он не существует) и помечает
                     его необработанным + поколение устанавливается равным (child_generation, если указано, иначе generation + 1).
                   - Создаёт связь FOLLOWING между user_id и sub_id.

                Args:
                    user_id (int): ID пользователя.
                    subscriptions (np.array): Массив ID пользователей, на которых подписан user_id.
                    create_followings (bool, optional): Определяет, нужно ли создавать узлы подписок.
                                                        По умолчанию True.
                    generation (int, optional): Поколение добавляемого пользователя. По умолчанию 0.
                    child_generation (int | None, optional): Поколение подписчиков. Если None, используется
                                                             generation + 1. По умолчанию None.

                Returns:
                    None
                """
        async with (await self.get_session()) as session:
            await session.run("""
                            MERGE (u:User {id: $id})
                            SET u.needToProcess = $flag, u.generation = $generation
                            """,
                              {"id": user_id, "flag": False, "generation": generation})

            if create_followings:
                await session.run("""
                UNWIND $subscriptions AS sub_id
                MERGE (s:User {id: sub_id})
                ON CREATE SET s.needToProcess = $flag, s.generation = $generation
                """, {
                    "subscriptions": subscriptions,
                    "flag": not (np.all(subscriptions == [-1]) or np.all(subscriptions == [-2])),
                    "generation": child_generation if child_generation is not None else generation + 1
                })

            await session.run("""
            UNWIND $subscriptions AS sub_id
            MATCH (u:User {id: $user_id}), (s:User {id: sub_id})
            MERGE (u)-[:FOLLOWING]->(s)
            """, {
                "subscriptions": subscriptions,
                "user_id": user_id
            })

    async def add_users(self, users_id: np.array, users_subscriptions: np.array, create_followings: bool = True,
                        generation: int = 0, child_generation: int | None = None) -> None:
        n_users = len(users_id)
        child_generation = generation + 1 if child_generation is None else child_generation

        flat_data = []
        subscr_data = []
        for i in range(n_users):
            user_id = users_id[i]
            subscriptions = users_subscriptions[i]

            sub_flag = not (np.all(subscriptions == [-1]) or np.all(subscriptions == [-2]))

            flat_data.extend([{
                'user_id': user_id,
                'sub_id': sub_id,
            } for sub_id in subscriptions])

            subscr_data.extend([{'sub_id': sub_id,
                                 'sub_flag': sub_flag,
                                 'child_generation': child_generation}
                                for sub_id in subscriptions])

        async with (await self.get_session()) as session:
            await session.run("""
                            UNWIND $users AS user_id
                            MERGE (u:User {id: user_id})
                            SET u.needToProcess = $flag, u.generation = $generation
                            """,
                              {"users": users_id, "flag": False, "generation": generation})

            if create_followings:
                await session.run("""
                    UNWIND $subscriptions_data AS sd
                    MERGE (s:User {id: sd.sub_id})
                    ON CREATE SET s.needToProcess = sd.sub_flag,
                                 s.generation = sd.child_generation
                    """, {"subscriptions_data": subscr_data})

            await session.run("""
                UNWIND $subscriptions_data AS sd
                MATCH (u:User {id: sd.user_id}), (s:User {id: sd.sub_id})
                MERGE (u)-[:FOLLOWING]->(s)
                """, {"subscriptions_data": flat_data})

    async def get_needed_to_process(self, generation: Optional[int] = None) -> pd.DataFrame:
        """
                Получает пользователей, которые помечены как требующие обработки (needToProcess=True).
                Args:
                    generation (Optional[int], optional): Фильтр по поколению пользователей.
                                                          Если None, возвращаются пользователи всех поколений.
                                                          По умолчанию None.
                Returns:
                    pd.DataFrame: Таблица с одним столбцом `user_id`, содержащая ID пользователей,
                                  которые требуют обработки.
                """
        async with (await self.get_session()) as session:
            if generation is None:
                result = await session.run("""
                        MATCH (u:User)
                        WHERE u.needToProcess = True
                        RETURN u.id AS user_id
                        """)
            else:
                result = await session.run("""
                                        MATCH (u:User)
                                        WHERE u.needToProcess = True AND u.generation = $generation
                                        RETURN u.id AS user_id
                                        """, {"generation": generation})
            return pd.DataFrame([record for record in await result.data()])

    async def get_all_users(self) -> pd.DataFrame:
        async with (await self.get_session()) as session:
            result = await session.run("""
                        MATCH (u:User)
                        RETURN u.id AS user_id
                        """)
            return pd.DataFrame([record for record in await result.data()])

    async def __create_users(self, users: list[User], generation: int) -> None:
        async with (await self.get_session()) as session:
            for user in users:
                await session.run("""
                        MERGE (u:User {id: $id})
                        SET u.name = $name, u.meta = $meta, u.generation = $generation
                        """,
                                  {"id": user.id, "name": user.name,
                                   "meta": user.metaInfo, "generation": generation})

    async def __create_subscriptions(self, users: list[User]) -> None:
        async with (await self.get_session()) as session:
            for user in users:
                for subscribe_on_id in user.subscriptions:
                    await session.run("""
                            MATCH (u:User {id: $user_id}), (f:User {id: $subscribe_on_id})
                            MERGE (u)-[:FOLLOWING]->(f)
                            """,
                                      {"user_id": user.id, "subscribe_on_id": subscribe_on_id})

    async def create_all(self, users: list[User], generation: int = 0) -> None:
        """
                Создаёт пользователей и их подписки в базе данных.
                В случае ошибки выводит сообщение об исключении.

                Args:
                    users (list[User]): Список объектов пользователей.
                    generation (int, optional): Поколение создаваемых пользователей. По умолчанию 0.

                Returns:
                    None
                """
        try:
            await self.__create_users(users, generation)
            await self.__create_subscriptions(users)
        except Exception as e:
            print(e)

    async def get_subscriptions(self, user_id: int) -> pd.DataFrame:  # just example ToDo
        """
                Получает список подписок (пользователей, на которых подписан указанный пользователь).

                Args:
                    user_id (int): ID пользователя, для которого ищутся подписки.

                Returns:
                    pd.DataFrame: Таблица с тремя столбцами:
                        - `following_id` (str): ID подписанного пользователя.
                        - `following_name` (str): Имя подписанного пользователя.
                        - `following_info` (any): Метаданные о подписанном пользователе.
                """
        async with (await self.get_session()) as session:
            result = await session.run("""
                    MATCH (u:User {id: $id})-[:FOLLOWING]->(f)
                    RETURN f.id AS following_id, f.name AS following_name, f.meta AS following_info
                    """,
                                       {"id": user_id})
            return pd.DataFrame([record for record in await result.data()])

    async def add_info_about_user(self, users_data: typing.List[typing.Dict]):
        async with (await self.get_session()) as session:
            await session.run("""
                    UNWIND $users AS user
                    MERGE (u:User {id: user.id})
                    SET u.screen_name = user.screen_name,
                        u.description = user.description,
                        u.is_blue_verified = user.is_blue_verified,
                        u.followers_count = user.followers_count,
                        u.name = user.name,
                        u.professional_type = user.professional_type,
                        u.statuses_count = user.statuses_count,
                        u.listed_count = user.listed_count,
                        u.verified_type = user.verified_type,
                        u.friends_count = user.friends_count,
                        u.created_at = user.created_at,
                        u.media_count = user.media_count
                                       """, {"users": users_data})


def process_user_subscriptions(directory_path: str):
    path = Path(directory_path)
    json_files = [f for f in path.glob('*.json') if f.is_file()]
    users = []
    datas = []

    for i, file_path in enumerate(json_files):
        if i == 100:
            break
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            user_id = file_path.stem
            users.append(user_id)
            datas.append(data)

        except json.JSONDecodeError:
            print(f"Ошибка чтения JSON в файле {file_path}")
        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {str(e)}")
    return users, datas


async def main():
    neo = Neo4jDAO()
    await neo.add_users(*process_user_subscriptions("E:/neo4j/friends_data"))
    await neo.close()


if __name__ == "__main__":
    asyncio.run(main())
