import asyncio
import os

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
        self.driver = AsyncGraphDatabase.driver(URI, auth=AUTH)

    async def close(self):
        await self.driver.close()

    async def get_session(self) -> AsyncSession:
        return self.driver.session(database="neo4j")

    async def add_empty_user(self, user_id: str, overwrite: bool = False):
        async with (await self.get_session()) as session:
            if overwrite:
                await session.run("""
                                            MERGE (u:User {id: $id})
                                            SET u.needToProcess = $flag
                                            """,
                                  {"id": user_id, "flag": True})
            else:
                await session.run("""
                                            MERGE (u:User {id: $id})
                                            ON CREATE SET u.needToProcess = $flag
                                            """,
                                  {"id": user_id, "flag": True})

    async def add_user(self, user_id: str, subscriptions: np.array, create_followers: bool = True) -> None:
        #
        async with (await self.get_session()) as session:
            await session.run("""
                            MERGE (u:User {id: $id})
                            SET u.needToProcess = $flag
                            """,
                              {"id": user_id, "flag": False})

            for sub_id in subscriptions:
                if create_followers:
                    await session.run("""
                    MERGE (s:User {id: $sub_id})
                    ON CREATE SET s.needToProcess = $flag
                    """, {"sub_id": sub_id, "flag": True})

                await session.run("""
                MATCH (u:User {id: $user_id}), (s:User {id: $sub_id})
                MERGE (u)-[:FOLLOWING]->(s)
                """, {"user_id": user_id, "sub_id": sub_id})

    async def get_needed_to_process(self) -> pd.DataFrame:
        async with (await self.get_session()) as session:
            result = await session.run("""
            MATCH (u:User)
            WHERE u.needToProcess = True
            RETURN u.id AS user_id
            """)
            return pd.DataFrame([record for record in await result.data()])

    async def __create_users(self, users: list[User]) -> None:
        async with (await self.get_session()) as session:
            for user in users:
                await session.run("""
                MERGE (u:User {id: $id})
                SET u.name = $name, u.meta = $meta
                """,
                                  {"id": user.id, "name": user.name, "meta": user.metaInfo})

    async def __create_subscriptions(self, users: list[User]) -> None:
        #
        async with (await self.get_session()) as session:
            for user in users:
                for subscribe_on_id in user.subscriptions:
                    await session.run("""
                    MATCH (u:User {id: $user_id}), (f:User {id: $subscribe_on_id})
                    MERGE (u)-[:FOLLOWING]->(f)
                    """,
                                      {"user_id": user.id, "subscribe_on_id": subscribe_on_id})

    async def create_all(self, users: list[User]) -> None:
        try:
            await self.__create_users(users)
            await self.__create_subscriptions(users)
        except Exception as e:
            print(e)

    async def get_subscriptions(self, user_id: str) -> pd.DataFrame:  # just example ToDo
        async with (await self.get_session()) as session:
            result = await session.run("""
            MATCH (u:User {id: $id})-[:FOLLOWING]->(f)
            RETURN f.id AS following_id, f.name AS following_name, f.meta AS following_info
            """,
                                       {"id": user_id})
            return pd.DataFrame([record for record in await result.data()])


async def main():
    users = load_users("example.csv")
    neo = Neo4jDAO()
    await neo.create_all(users)
    print("Ok1\n")

    followers = await neo.get_subscriptions("2")
    print(followers)
    print("\nOk2\n")

    await neo.add_user("4", ["1", "3"])
    await neo.add_user("5", ["100"], False)
    await neo.add_user("6", ["101"], True)
    print("\nOk3\n")

    print(await neo.get_needed_to_process())
    print("Ok4")

    await neo.close()


if __name__ == "__main__":
    asyncio.run(main())
