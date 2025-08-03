import os

import pandas as pd
from graph_tool.all import Graph, minimize_blockmodel_dl

def bulid_graph():
    df_edges = pd.read_csv("graph/all_edges.csv")
    gt_graph = Graph(directed=True)
    node_map = {}
    reverse_node_map = {}
    i = -1
    for _, row in df_edges.iterrows():
        for node_id in (row['src'], row['dst']):
            if node_id not in node_map:
                i += 1
                node_map[node_id] = gt_graph.add_vertex()
                reverse_node_map[i] = node_id

        gt_graph.add_edge(node_map[row['src']], node_map[row['dst']])
    return gt_graph, reverse_node_map

id_to_node_id = pd.read_csv("graph/id_to_node_id.csv")
gt_graph, reverse_node_map = None, None

def sbm(k:int = 25):
    if os.path.exists(f'graph/sbm/{k}.csv'):
        df = pd.read_csv(f'graph/sbm/{k}.csv')
        final_df = pd.merge(
            left=df,
            right=id_to_node_id,
            left_on='user_id',
            right_on='node_id',
            how='left'
        )
        df["user_id"] = final_df["user_id_1"]
        return df
    global gt_graph, reverse_node_map
    if gt_graph is None:
        gt_graph, reverse_node_map = bulid_graph()
    state = minimize_blockmodel_dl(gt_graph, multilevel_mcmc_args={"B_max": k, "B_min": k})
    blocks = state.get_blocks()

    data = []
    for v in gt_graph.vertices():
        user_id = reverse_node_map[v]
        result_community = blocks[v]
        data.append((user_id, result_community))
    df = pd.DataFrame(data, columns=['user_id', 'result_community'])
    final_df = pd.merge(
        left=df,
        right=id_to_node_id,
        left_on='user_id',
        right_on='node_id',
        how='left'
    )
    df["user_id"] = final_df["user_id_1"]
    os.makedirs("graph/sbm", exist_ok=True)
    df.to_csv(f"graph/sbm/{k}.csv", index=False)
    return df
