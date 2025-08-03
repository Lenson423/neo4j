import asyncio
import os
from enum import Enum
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
import pandas as pd


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


    @staticmethod
    def get_property_name(edge_type: Edge):
        property_name = "needToProcess"
        if edge_type != Edge.FOLLOWING:
            property_name += str(edge_type)
        return property_name

    async def louvain(self, session,  name:str = 'usersGraph', max_levels: int = 20,
                      tolerance: float = 0.00001, min_size: int = 15) -> pd.DataFrame:
        result = await session.run("""
                CALL gds.louvain.stream($graph_name, {
                    includeIntermediateCommunities: true,
                    maxLevels: $max_levels,
                    tolerance: $tolerance,
                    minCommunitySize: $min_community_size
                })
                YIELD nodeId, communityId, intermediateCommunityIds
                RETURN gds.util.asNode(nodeId).id AS user_id, communityId as result_community
                ORDER BY communityId
            """, {"graph_name": name, "max_levels": max_levels, "tolerance": tolerance, "min_community_size": min_size})
        return pd.DataFrame([record for record in await result.data()])

    async def maxkcut(self, session,  name:str = 'usersGraph', k:int = 20, iterations: int = 25) -> pd.DataFrame:
        if os.path.exists(f"graph/maxkcut/{name}.csv"):
            df = pd.read_csv(f"graph/maxkcut/{name}.csv")
            return df
        result = await session.run("""
                CALL gds.maxkcut.stream($graph_name, {
                    k: $k,
                    iterations: $iterations
                })
                YIELD nodeId, communityId
                RETURN gds.util.asNode(nodeId).id AS user_id, communityId as result_community
                ORDER BY communityId
            """, {"graph_name": name, "k": k, "iterations": iterations})
        res = pd.DataFrame([record for record in await result.data()])
        os.makedirs("graph/maxkcut", exist_ok=True)
        res.to_csv(f"graph/maxkcut/{name}.csv", index=False)
        return res
