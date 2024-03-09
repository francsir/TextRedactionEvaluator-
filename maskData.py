from datasets import load_dataset, Dataset
from transformers import DataCollatorForLanguageModeling
import collections
import numpy as np
import random
import string


def group_texts(dataset):
    chunk_size = 64

    concatenated_text = ''.join(dataset['text'])

    chunks_text = []
    current_chunk = ""

    for word in concatenated_text.split():
        if len(current_chunk) + len(word) <= chunk_size:
            current_chunk += word + " "
        else:
            chunks_text.append(current_chunk.strip())
            current_chunk = word + " "

    if current_chunk:
        chunks_text.append(current_chunk.strip())

    chunks = {
        'text': chunks_text
    }
    chunks['label'] = chunks['text'].copy()
    chunks_ds = Dataset.from_dict(chunks)

    return chunks_ds


##def group_texts(dataset):
##    chunk_size = 64
##
##    concatenated_text = ''.join(dataset['text'])
##
##    total_length = len(concatenated_text)
##    total_length = (total_length // chunk_size) * chunk_size
##    
##    chunks = {
##        'text': [concatenated_text[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
##    }
##    chunks['label'] = chunks['text'].copy()
##    chunks_ds = Dataset.from_dict(chunks)
##    
##    return chunks_ds

def remove_punc(word):
    translation_table = str.maketrans("", "", string.punctuation)
    return word.translate(translation_table)

def mask_random_words(dataset, mask_count):
    texts = []
    originals = []
    for i in range(len(dataset['text'])):

        tokens = dataset['text'][i].split()
        available_ind = list(range(len(tokens)))

        original = dataset['text'][i]
        if tokens:
            
            for _ in range(mask_count):

                if not available_ind:
                    break

                bad_word = True
                i = 0
                while(bad_word): 
                    idx_to_mask = random.choice(available_ind)
                    

                    masked_word = tokens[idx_to_mask]

                    no_punc = remove_punc(masked_word)

                    if no_punc != '':
                        bad_word = False
                        available_ind.remove(idx_to_mask)
                        i += 1
                    if i > len(tokens):
                        continue
                

                tokens[idx_to_mask] = '[MASK]'
            texts.append(" ".join(tokens))
            originals.append(original)
                
            
    masked = Dataset.from_dict({'text': texts, 'original': originals})

    return masked
            
    
 
def __main__(mask_count = 2, size = 3):
    
    ## load dataset
    dataset = load_dataset("imdb")
    
    ## Split into chunks
    chunks = group_texts(dataset['unsupervised'][:size])

    ## Mask mask_count amount of words
    masked_data = mask_random_words(chunks, mask_count)

    ## Save
    masked_data.save_to_disk(f"./datasets/imdb_{mask_count}_{size}")
    print("done")


