import json
from collections import Counter

with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

counter = Counter(entry["communityId"] for entry in data)

for community_id, count in counter.items():
    print(f"Community ID {community_id}: {count} записей")
