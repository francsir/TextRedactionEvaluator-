from datasets import load_dataset, Dataset
from transformers import DataCollatorForLanguageModeling
import collections
import numpy as np
import random
import string





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

def remove_punc(word):
    translation_table = str.maketrans("", "", string.punctuation)
    return word.translate(translation_table)

def mask_random_words(dataset):
    texts = []
    label = []
    for i in range(len(dataset['text'])):

        tokens = chunks['text'][i].split()
        

        if tokens:
            bad_word = True
            i = 0
            while(bad_word): 
                idx_to_mask = random.randint(0, len(tokens)-1)
            
                masked_word = tokens[idx_to_mask]

                no_punc = remove_punc(masked_word)

                if no_punc != '':
                    bad_word = False
                    i += 1
                if i > len(tokens):
                    continue
                

            tokens[idx_to_mask] = '[MASK]'
            texts.append(" ".join(tokens))
            label.append(no_punc)
    
    masked = Dataset.from_dict({'text': texts, 'label': label})

    return masked
            
    
 

dataset = load_dataset("imdb")
chunks = group_texts(dataset['unsupervised'])
masked_data = mask_random_words(chunks)

masked_data.save_to_disk('./datasets/imdb')



