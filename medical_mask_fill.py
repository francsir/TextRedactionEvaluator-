from transformers import AutoTokenizer, AutoModel, pipeline
import pandas as pd
import concurrent.futures



dataset = pd.read_csv('./medical_tc_test_masked_medical_3.csv')
classifier = pipeline("fill-mask", model="medicalai/ClinicalBert")

def process_chunk(chunk):
    count = chunk.count('[MASK]')
    print(1)
    for i in range(count):
        temp = classifier(chunk)
        try:
            chunk = temp[0]['sequence']
        except:
            chunk = temp[0][0]['sequence']
    return chunk



def predict_sentence(text, i):
    ##get the first prediction
    i = i+1
    print(i)
    if len(text) > 512:
        chunks = [text[i:i + 512] for i in range(0, len(text), 512)]
    else:
        chunks = [text]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_chunk, chunks))

    return ' '.join(results)


i = 0
dataset['predicted'] = dataset['masked'].apply(predict_sentence(i))


dataset.to_csv(f'medicalData\Medical-Abstracts-TC-Corpus\medical_tc_test_masked_medical_4.csv', index=False)