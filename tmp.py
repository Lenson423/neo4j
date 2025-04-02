from transformers import pipeline

model_name = 'facebook/bart-large-mnli'
classifier = pipeline('zero-shot-classification', model=model_name)

classifier.model.save_pretrained('./local_model')
classifier.tokenizer.save_pretrained('./local_model')