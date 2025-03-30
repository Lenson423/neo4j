import numpy as np
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str
    metaInfo: str #description or something else ToDo
    subscriptions: np.array
