import asyncio
import logging
from logging.handlers import RotatingFileHandler

import pandas as pd
import os
from dotenv import load_dotenv
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession

logger = logging.getLogger()
logger.setLevel(logging.INFO)
os.makedirs("logs", exist_ok=True)
file_handler = RotatingFileHandler(
    'logs/app.log',
    maxBytes=1024*1024*64,
    backupCount=3
)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

load_dotenv()
URI = os.getenv("URI")
AUTH = (os.getenv("AUTH_NAME"), os.getenv("AUTH_PASSWORD"))

async def create_crypto_graph(session, graph_name):
    logging.log(logging.INFO, f"Creating graph {graph_name}")
    await session.run(f"""
        CALL gds.graph.project.cypher(
            $graph_name,
            'MATCH (u:User) WHERE u.is_crypto_related = true RETURN id(u) AS id',
            'MATCH (u1:User)-[:FOLLOWING]->(u2:User)
             WHERE u1.is_crypto_related = true AND u2.is_crypto_related = true
             RETURN id(u1) AS source, id(u2) AS target'
        )
    """, {"graph_name": graph_name})
    logging.log(logging.INFO, f"Graph created {graph_name}")

async def ensure_graph_exists(session, graph_name):
    result = await session.run("""
        CALL gds.graph.exists($name) YIELD exists
        RETURN exists
    """, {"name": graph_name})
    record = await result.single()
    if not record["exists"]:
        await create_crypto_graph(session, graph_name)

async def create_subgraph(session, subgraph_name, node_ids_str):
    result = await session.run("""
            CALL gds.graph.exists($name) YIELD exists
            RETURN exists
        """, {"name": subgraph_name})
    record = await result.single()
    if record["exists"]:
        return

    logging.log(logging.INFO, f"Start creating subgraph {subgraph_name}")
    await session.run(f"""
        CALL gds.graph.project.cypher(
            $subgraph_name,
            'MATCH (u:User) WHERE u.is_crypto_related = true AND u.id IN {node_ids_str}
             RETURN id(u) AS id',
            'MATCH (u1:User)-[:FOLLOWING]->(u2:User)
             WHERE u1.is_crypto_related = true AND u1.id IN {node_ids_str}
             AND u2.is_crypto_related = true AND u2.id IN {node_ids_str}
             RETURN id(u1) AS source, id(u2) AS target'
        )
    """, {
        "subgraph_name": subgraph_name
    })
    logging.log(logging.INFO, f"End creating subgraph {subgraph_name}")


class GraphAnalyzer:

    driver: AsyncDriver

    def __init__(self):
        self.driver = AsyncGraphDatabase.driver(URI, auth=AUTH, max_connection_pool_size=20)


    async def close(self):
        await self.driver.close()

    async def get_session(self) -> AsyncSession:
        return self.driver.session(database="neo4j")

    async def run_label_propagation(self, session, graph_name, max_iterations: int = 20) -> pd.DataFrame:
        logging.log(logging.INFO, f"Start label_propagation on {graph_name}")
        result = await session.run("""
            CALL gds.labelPropagation.stream($graph_name, {
                maxIterations: $max_iterations
            })
            YIELD nodeId, communityId
            RETURN gds.util.asNode(nodeId).id AS user_id, communityId
            ORDER BY communityId
        """, {"graph_name": graph_name, "max_iterations": max_iterations})
        logging.log(logging.INFO, f"End label_propagation on {graph_name}")
        return pd.DataFrame([record for record in await result.data()])

    async def run_modularity_on_component(self, session,
                                          graph_name, component_id, component_nodes,
                                          max_iterations: int = 20, tolerance:float = 1e-6,
                                          min_community_size: int = 15) -> pd.DataFrame:
        subgraph_name = f"{graph_name}_component_{component_id}"

        if not all(isinstance(x, int) for x in component_nodes):
            raise ValueError("All component_nodes must be integers.")

        node_ids_str = "[" + ", ".join(map(str, component_nodes)) + "]"
        await create_subgraph(session, subgraph_name, node_ids_str)

        logging.log(logging.INFO, f"Start modularity on {subgraph_name}")

        result = await session.run("""
            CALL gds.modularityOptimization.stream($subgraph_name, {
                maxIterations: $max_iterations,
                tolerance: $tolerance,
                minCommunitySize: $min_community_size
            })
            YIELD nodeId, communityId
            RETURN gds.util.asNode(nodeId).id AS user_id, communityId as result_community
            ORDER BY communityId
        """, {"subgraph_name": subgraph_name, "max_iterations": max_iterations,
              "tolerance": tolerance, "min_community_size": min_community_size})
        df = pd.DataFrame([record for record in await result.data()])
        logging.log(logging.INFO, f"End modularity on {subgraph_name}")

        logging.log(logging.INFO, f"Start deleting subgraph {subgraph_name}")
        await session.run("CALL gds.graph.drop($name)", {"name": subgraph_name})
        logging.log(logging.INFO, f"End deleting subgraph {subgraph_name}")
        return df

    async def analyze(self, session, graph_name, max_iterations: int = 20, tolerance:float = 1e-6,
                                          min_community_size: int = 15):
            await ensure_graph_exists(session, graph_name)

            df_wcc = await self.run_label_propagation(session, graph_name, max_iterations)

            all_dfs = []
            for component_id, group in df_wcc.groupby("communityId"):
                if len(group) >= 15:
                    df_modularity = await self.run_modularity_on_component(
                        session, graph_name, component_id, group["user_id"].tolist(),
                        max_iterations, tolerance, min_community_size
                    )
                    all_dfs.append(df_modularity)

            final_df = pd.concat(all_dfs, ignore_index=True)
            return final_df

async def main():
    neo = GraphAnalyzer()
    data = await neo.analyze()
    os.makedirs("results", exist_ok=True)
    data.to_csv("results/wtf.csv", index=False)
    await neo.close()


if __name__ == "__main__":
    asyncio.run(main())
