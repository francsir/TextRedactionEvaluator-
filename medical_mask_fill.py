from transformers import AutoTokenizer, AutoModel, pipeline
import pandas as pd



dataset = pd.read_csv('medicalData\Medical-Abstracts-TC-Corpus\medical_tc_test_masked_medical_3.csv')


#tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
#model = AutoModel.from_pretrained("medicalai/ClinicalBERT")

classifier = pipeline("fill-mask", model="medicalai/ClinicalBert")

def predict_sentence(text):
    ##get the first prediction
    trunc_text = []

    if len(text) > 512:
        chunks = [text[i:i + 512] for i in range(0, len(text), 512)]
    else:
        chunks = [text]

    for chunk in chunks:
        count = chunk.count('[MASK]')
        for i in range(count):
            temp = classifier(chunk)
            try:
                chunk = temp[0]['sequence']
            except:
                chunk = temp[0][0]['sequence']
            
        trunc_text.append(chunk)

    return ' '.join(trunc_text)


dataset['predicted'] = dataset['masked'].apply(predict_sentence)


dataset.to_csv(f'medicalData\Medical-Abstracts-TC-Corpus\medical_tc_test_masked_medical_4.csv', index=False)