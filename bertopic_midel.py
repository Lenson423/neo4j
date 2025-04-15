import json

import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import umap
import hdbscan

with open("users_comments.json", "r", encoding="utf-8") as f:
    data = json.load(f)

user_ids = []
user_texts = []

for user in data:
    user_id = user["user_id"]
    comments = user["comments"]
    full_text = " ".join(comments).strip()
    if full_text:
        user_ids.append(user_id)
        user_texts.append(full_text)

print(f"[INFO] Загружено пользователей: {len(user_ids)}")

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

topics, probs = topic_model.fit_transform(user_texts)
topic_model.save("my_model")


def get_topic_keywords(topic_id, top_n=5):
    if topic_id == -1:
        return ["unknown"]
    return [word for word, _ in topic_model.get_topic(topic_id)[:top_n]]


user_tags = {
    user_id: get_topic_keywords(topic_id)
    for user_id, topic_id in zip(user_ids, topics)
}

df_result = pd.DataFrame({
    "user_id": user_ids,
    "topic_id": topics,
    "tags": [", ".join(tags) for tags in user_tags.values()]
})

df_result.to_csv("user_tags.csv", index=False, encoding="utf-8-sig")
print("[INFO] Результаты сохранены в user_tags.csv")
