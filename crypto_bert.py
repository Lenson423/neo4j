import pandas as pd
from datasets import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

tokenizer = BertTokenizer.from_pretrained("kk08/CryptoBERT")
model = BertForSequenceClassification.from_pretrained("kk08/CryptoBERT")

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def is_crypto_related(description, screen_name, name):
    user_info = f"{description}; {name}; {screen_name}"
    result = classifier(user_info)
    return result[0]['label'] == 'LABEL_1'


data = pd.read_csv('data.csv')
results = []

for _, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing users"):
    result = is_crypto_related(row['description'], row['screen_name'], row['name'])
    results.append(result)

data['is_crypto_related'] = results
data.to_csv('result.csv', index=False)
