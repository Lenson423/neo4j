import pandas as pd
from tqdm import tqdm
from transformers import pipeline

classifier = pipeline('zero-shot-classification', model='./local_model', tokenizer='./local_model')


def is_crypto_related(description, screen_name, name):
    user_info = f"{description}; {name}; {screen_name}"
    candidate_labels = ['crypto', 'non-crypto']
    result = classifier(user_info, candidate_labels)
    return result['labels'][0] == 'crypto'


data = pd.read_csv('data.csv')
results = []

for _, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing users"):
    result = is_crypto_related(row['description'], row['screen_name'], row['name'])
    results.append(result)

data['is_crypto_related'] = results
data.to_csv('result.csv', index=False)
