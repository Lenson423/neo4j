import csv
import numpy as np
from userInfo import User


def load_users(file_path: str = "example.csv") -> list[User]:
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        users = []
        for row in reader:
            user_id = row["id"]
            name = row["name"]
            meta = row["meta"]
            subscriptions = np.array(list(map(int, row["subscriptions"].split(",") if row["subscriptions"] else [])))
            users.append(User(int(user_id), name, meta, subscriptions))
        return users
