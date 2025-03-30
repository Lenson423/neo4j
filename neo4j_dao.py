import os

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver, Session
import pandas as pd

from csv_processor import load_users
from userInfo import User

load_dotenv()
URI = os.getenv("URI")
AUTH = (os.getenv("AUTH_NAME"), os.getenv("AUTH_PASSWORD"))


class Neo4jDAO:  # ToDO
    driver: Driver

    def __init__(self):
        self.driver = GraphDatabase.driver(URI, auth=AUTH)
        self.driver.verify_connectivity()

    def __del__(self):
        self.driver.close()

    def get_session(self) -> Session:
        return self.driver.session(database="neo4j")

    def __create_users(self, users: list[User]) -> None:
        with self.get_session() as session:
            for user in users:
                session.run("""
                MERGE (u:User {id: $id})
                SET u.name = $name, u.meta = $meta
                """,
                            {"id": user.id, "name": user.name, "meta": user.metaInfo})

    def __create_subscriptions(self, users: list[User]) -> None:
        with self.get_session() as session:
            for user in users:
                for follower_id in user.followers:
                    session.run("""
                    MATCH (u:User {id: $user_id}), (f:User {id: $follower_id})
                    MERGE (u)-[:FOLLOWS]->(f)
                    """,
                                {"user_id": user.id, "follower_id": follower_id})

    def create_all(self, users: list[User]) -> None:
        try:
            self.__create_users(users)
            self.__create_subscriptions(users)
        except Exception as e:
            print(e)

    def get_followers(self, user_id: str) -> pd.DataFrame:  # just example ToDo
        with self.get_session() as session:
            result = session.run("""
            MATCH (u:User {id: $id})-[:FOLLOWS]->(f)
            RETURN f.id AS follower_id, f.name AS follower_name, f.meta AS follower_info
            """,
                                 {"id": user_id})
            return result.to_df()


if __name__ == "__main__":
    users = load_users("example.csv")
    neo = Neo4jDAO()
    neo.create_all(users)
    print("Ok1\n")

    followers = neo.get_followers("2")
    print(followers)
    print("\nOk2")
