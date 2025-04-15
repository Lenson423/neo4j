import json

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import umap
import hdbscan

vectorizer_model = CountVectorizer(
    stop_words="english",
    max_features=30000,
    ngram_range=(1, 2)
)

umap_model = umap.UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=15,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

topic_model = BERTopic(
    language="english",
    embedding_model="all-MiniLM-L6-v2",
    vectorizer_model=vectorizer_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    top_n_words=10,
    calculate_probabilities=False,
    verbose=True,
    low_memory=True,
)


def get_comments(path: str, basic: str = "/home/ubuntu/tweets/"):
    with open(basic + path + '.json', "r") as file:
        data = json.load(file)
        if data is not None:
            return data.values()
        else:
            return [""]


texts_split = []
for doc in get_comments("bitcatshow"):
    texts_split.append(str(doc))

topic_model.reduce_topics(texts_split, nr_topics=50)
topic_model.save("my_model")
topics, probs = topic_model.fit_transform(texts_split)
print(topics)
print()
print(topic_model.get_topic_info())
