from transformers import AutoTokenizer, AutoModel, pipeline
import pandas as pd
import concurrent.futures



dataset = pd.read_csv('./medical_tc_test_masked_medical_3.csv')
classifier = pipeline("fill-mask", model="medicalai/ClinicalBert")
global i
i = 0
def process_chunk(chunk):
    count = chunk.count('[MASK]')
    for i in range(count):
        temp = classifier(chunk)
        try:
            chunk = temp[0]['sequence']
        except:
            chunk = temp[0][0]['sequence']
    return chunk



def predict_sentence(text):
    ##get the first prediction
    global i 
    i = i+1
    print(i)
    if len(text) > 512:
        chunks = [text[i:i + 512] for i in range(0, len(text), 512)]
    else:
        chunks = [text]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_chunk, chunks))

    return ' '.join(results)


dataset['predicted'] = dataset['masked'].apply(predict_sentence)


dataset.to_csv(f'medicalData\Medical-Abstracts-TC-Corpus\medical_tc_test_masked_medical_4.csv', index=False)