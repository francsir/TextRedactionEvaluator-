import pandas as pd 
import random 
import xml.etree.ElementTree as ET
import re

def words_from_xml():
    tree = ET.parse('./desc2024.xml')
    root = tree.getroot()

    unique_words = set()
    for descriptor_record in root.findall('.//DescriptorRecord'):
        for descriptor_name in descriptor_record.findall('.//DescriptorName'):
            string_element = descriptor_name.find('String')
            if string_element is not None:
                words = string_element.text.lower().split()
                unique_words.update(words)
    return list(unique_words)



def mask_random_words(text, n):
    words = text.split()

    if n > len(words):
        return '[MASK]' * len(words)

    
    indices = random.sample(range(len(words)), n)
    masked_words = [word if idx not in indices else "[MASK]" for idx, word in enumerate(words)]

    return " ".join(masked_words)

def mask_medical_words(text, medical_words):
    words = text.split()

    for i, word in enumerate(words):
        if word in medical_words:
            words[i] = '[MASK]'

    return " ".join(words)

def mask(medical_words, df):
    pattern = r'\b(?:{})\b'.format('|'.join(map(re.escape, medical_words)))

    df['masked'] = df['medical_abstract'].apply(lambda x: re.sub(pattern, ' [MASK] ', x, flags=re.IGNORECASE))
    df['masked'] = df['masked'].apply(lambda x: re.sub(r'\s+', ' ', x))

    return df





#load words from txt file

with open('medicalData\Medical-Abstracts-TC-Corpus\medical_words.txt', 'r') as file:
    words = file.read().splitlines()



dataset = pd.read_csv('medicalData\Medical-Abstracts-TC-Corpus\medical_tc_test.csv')
dataset = mask(words, dataset)
#dataset['masked'] = dataset['medical_abstract'].apply(lambda x: mask_medical_words(x, words))

#n = 50
#dataset['masked'] = dataset['medical_abstract'].apply(lambda x: mask_random_words(x, n))

dataset.to_csv(f'medicalData\Medical-Abstracts-TC-Corpus\medical_tc_test_masked_medical_3.csv', index=False)