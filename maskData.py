from datasets import load_dataset, Dataset
from transformers import DataCollatorForLanguageModeling
import collections
import numpy as np
import random





def group_texts(dataset):
    chunk_size = 64

    concatenated_text = ''.join(dataset['text'])

    total_length = len(concatenated_text)
    total_length = (total_length // chunk_size) * chunk_size
    
    chunks = {
        'text': [concatenated_text[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
    }
    chunks['label'] = chunks['text'].copy()
    chunks_ds = Dataset.from_dict(chunks)
    
    return chunks_ds

def mask_random_words(dataset):
    texts = []
    label = []
    for i in range(len(dataset['text'])):
        tokens = chunks['text'][i].split()

        if tokens: 
            idx_to_mask = random.randint(0, len(tokens)-1)
            masked_word = tokens[idx_to_mask]

            

            tokens[idx_to_mask] = '[MASK]'
            texts.append(" ".join(tokens))
            label.append(masked_word)
    
    masked = Dataset.from_dict({'text': texts, 'label': label})

    return masked
            
    
 

dataset = load_dataset("imdb")
chunks = group_texts(dataset['unsupervised'][:10])


masked_data = mask_random_words(chunks)

masked_data.save_to_disk('./datasets/imdb')



