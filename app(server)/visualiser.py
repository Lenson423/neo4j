import json
import logging
from logging.handlers import RotatingFileHandler

from bs4 import BeautifulSoup
from neo4j import GraphDatabase, basic_auth
from pandas import DataFrame
import pandas as pd
import graphistry
import matplotlib.pyplot as plt
from matplotlib import colors

import os
from dotenv import load_dotenv

logger = logging.getLogger()
logger.setLevel(logging.INFO)
os.makedirs("logs", exist_ok=True)
file_handler = RotatingFileHandler(
    'logs/visualiser.log',
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
driver = GraphDatabase.driver(URI, auth=AUTH, max_connection_pool_size=20)

AUTH_GRAPH = (os.getenv("GRAPH_NAME"), os.getenv("GRAPH_PASSWORD"))
print(AUTH_GRAPH)
graphistry.register(api=3, protocol='https', server='hub.graphistry.com',
                    personal_key_id = AUTH_GRAPH[0],
                    personal_key_secret=AUTH_GRAPH[1])

users = pd.read_csv("graph/nodes.csv")
mentions = pd.read_csv("graph/edges.csv")
id_to_screenname = pd.read_csv("graph/id_to_name.csv")

def visualise_graphistry(type: str, community: pd.DataFrame) -> (str, pd.DataFrame):
    logging.info("Start visualising Graphistry")
    if type == "pagerank":
        data = pd.read_csv("graph/pagerank.csv")
    elif type == "closeness":
        data = pd.read_csv("graph/closeness_centrality.csv")
    elif type == "eigenvector":
        data = pd.read_csv("graph/eigenvector_centrality.csv")
    else:
        raise Exception("Invalid type")
    valid_names = set(users['screen_name'])

    df = pd.DataFrame([{
    'screen_name': row['user_id'],
    'score': row['score']
} for _, row in data.iterrows() if row['user_id'] in valid_names])

    merged_step1 = pd.merge(
        left=community,
        right=id_to_screenname,
        left_on='user_id',
        right_on='user_id',
        how='left'
    )

    final_df = pd.merge(
        left=df,
        right=merged_step1,
        left_on='screen_name',
        right_on='screen_name',
        how='left'
    )

    final_df['result_community'] = pd.factorize(final_df['result_community'])[0]
    num_colors = max(final_df['result_community']) + 1
    cmap = plt.get_cmap('tab20b', num_colors)

    viz = graphistry.bind(source="u1", destination="u2", node="screen_name", point_size="score").nodes(final_df).edges(
        mentions).encode_point_color('result_community',
                                     categorical_mapping={i: colors.rgb2hex(cmap(i)) for i in range(num_colors)},
                                     default_mapping='orange')
    url = viz.plot()
    logging.info("End visualising Graphistry")

    example = pd.read_json("static/graph/my/modularity_with_score.json")
    merged_df = pd.merge(example, merged_step1, on='screen_name', how='left', suffixes=('_old', '_new'))
    example['community'] = merged_df['result_community']
    example['community'] = pd.factorize(example['community'])[0]
    return url, example