import asyncio
import json
import os
import random
import typing
from enum import Enum
from pathlib import Path
from typing import Optional

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
import pandas as pd
from typing_extensions import List

from csv_processor import load_users
from userInfo import User

load_dotenv()
URI = os.getenv("URI")
AUTH = (os.getenv("AUTH_NAME"), os.getenv("AUTH_PASSWORD"))


class Edge(Enum):
    REPLY = "REPLY"
    MENTIONS = "MENTIONS"
    RETWEETS = "RETWEETS"
    QUOTE_TWEETS = "QUOTE TWEETS"
    FOLLOWING = "FOLLOWING"

    def __str__(self):
        return self.value


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

    @staticmethod
    def get_property_name(edge_type: Edge):
        property_name = "needToProcess"
        if edge_type != Edge.FOLLOWING:
            property_name += str(edge_type)
        return property_name

    async def add_edges_typed(self, users_ids: List, users_subscriptions: List[List], edge_type: Edge) -> None:
        """
        :param users_ids: все пользователи уже должны быть в БД при вызове этого метода
        :param users_subscriptions: все пользователи уже должны быть в БД при вызове этого метода
        """
        if len(users_subscriptions) != len(users_ids):
            raise Exception('users_subscriptions and users_ids must have same length')

        flat_data = []
        for i in range(len(users_ids)):
            user_id = users_ids[i]
            subscriptions = users_subscriptions[i]

            flat_data.extend([{
                'user_id': user_id,
                'sub_id': sub_id,
                'edge_type': str(edge_type),
            } for sub_id in subscriptions])

        async with (await self.get_session()) as session:
            await session.run("""
                UNWIND $subscriptions_data AS sd
                MATCH (u:User {id: sd.user_id}), (s:User {id: sd.sub_id})
                CALL apoc.create.relationship(u, sd.edge_type, {}, s) YIELD rel
                RETURN u, s, rel
                """, {"subscriptions_data": flat_data})

            await session.run("""
                UNWIND $users AS user_id
                MERGE (u:User {id: user_id})
                WITH u
                CALL apoc.create.setProperty(u, $property, $flag) YIELD node
                RETURN node
                """,{"users": users_ids, "flag": False, "property": self.get_property_name(edge_type)})

    async def get_not_processed_users(self, edge_type) -> pd.DataFrame:
        async with (await self.get_session()) as session:
            result = await session.run("""
                MATCH (u:User)
                WITH u, apoc.any.property(u, $property) AS prop
                WHERE prop IS NULL OR prop = true
                RETURN u.id AS user_id
                """, {"property": self.get_property_name(edge_type)})
            return pd.DataFrame(await result.data())

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

    async def get_all_users_data(self) -> pd.DataFrame:
        async with (await self.get_session()) as session:
            result = await session.run("""
                        MATCH (u:User)
                        RETURN u.id AS user_id, u.description as description,
                        u.screen_name as screen_name, u.name as name
                        """)
            return pd.DataFrame([record for record in await result.data()])

    async def get_user_data(self, name: str) -> dict:
        async with (await self.get_session()) as session:
            result = await session.run("""
                        MATCH (u:User {screen_name: $user_name})
                        RETURN u.description as description,
                        u.screen_name as screen_name,
                        u.name as name
                        """, {"user_name": name})
            return (await result.data())[0]

    async def set_users_groups(self, user_names: list[str], groups: list[str]) -> None:
        if len(user_names) != len(groups):
            raise ValueError("Длины списков user_names и groups должны совпадать")
        subscriptions_data = [{
            "user_name": user_name,
            "topic_id": topic_id,
        } for idx, (user_name, topic_id) in enumerate(zip(user_names, groups))]

        async with (await self.get_session()) as session:
            await session.run("""
                    UNWIND $subscriptions_data AS sd
                    MERGE (u:User {screen_name: sd.user_nameZ})
                    SET u.topic = sd.topic_id
                """, {"subscriptions_data": subscriptions_data})

    async def get_all_followers(self, name) -> pd.DataFrame:
        async with (await self.get_session()) as session:
            result = await session.run("""
            MATCH (f:User)-[:FOLLOWING]->(u:User {screen_name: $user_name})
            RETURN f.screen_name as screen_name,
            f.name as name, f.description as description
            """, {"user_name": name})
            return pd.DataFrame([record for record in await result.data()])

    async def process_labels1(self, users: list, labels: list[str]):
        users_data = [{'id': id1, 'label': label} for id1, label in zip(users, labels)]
        async with (await self.get_session()) as session:
            await session.run("""
                    UNWIND $users AS user
                    MERGE (u:User {id: user.id})
                    SET u.type1 = user.label,
                                       """, {"users": users_data})

    async def create_graph(self):
        async with (await self.get_session()) as session:
            await session.run("""CALL gds.graph.project(
                                'usersGraph',
                                'User',
                                { FOLLOWING: { orientation: 'NATURAL' } }
                                );
                                """)

    async def wcc(self) -> pd.DataFrame:
        async with (await self.get_session()) as session:
            result = await session.run("""
        CALL gds.wcc.stream('usersGraph')
        YIELD nodeId, componentId
        RETURN gds.util.asNode(nodeId).id AS user_id, componentId
        ORDER BY componentId
        """)
            return pd.DataFrame([record for record in await result.data()])

    async def page_rank(self) -> pd.DataFrame:
        async with (await self.get_session()) as session:
            result = await session.run("""CALL gds.pageRank.stream('usersGraph')
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).screen_name AS user_id, score
                ORDER BY score DESC
                """)
            return pd.DataFrame([record for record in await result.data()])

    async def louvain(self) -> pd.DataFrame:
        async with (await self.get_session()) as session:
            result = await session.run("""
                CALL gds.louvain.stream('usersGraph', {
                    includeIntermediateCommunities: true,
                    maxLevels: 20,
                    tolerance: 0.00001
                })
                YIELD nodeId, communityId, intermediateCommunityIds
                RETURN gds.util.asNode(nodeId).id AS user_id, communityId, intermediateCommunityIds
                ORDER BY communityId
            """)
            return pd.DataFrame([record for record in await result.data()])


async def main():
    neo = Neo4jDAO()
    print(await neo.get_not_processed_users(Edge.QUOTE_TWEETS))
    print()
    await neo.add_empty_user(-3)
    await neo.add_empty_user(-4)
    await neo.add_empty_user(-5)

    await neo.add_edges_typed([-3, -5], [[-5], [-3, -4]], Edge.QUOTE_TWEETS)
    print(await neo.get_not_processed_users(Edge.QUOTE_TWEETS))
    await neo.close()


if __name__ == "__main__":
    asyncio.run(main())
