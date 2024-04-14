from transformers import AutoTokenizer, AutoModel, pipeline
import pandas as pd


global j
j = -1

output_dir = "model/clinical-trained"
classifier = pipeline("fill-mask", model=output_dir, tokenizer=output_dir)
dataset = pd.read_csv('./medical_tc_test_masked_medical_3.csv')

tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
#model = AutoModel.from_pretrained("medicalai/ClinicalBERT")

#classifier = pipeline("fill-mask", model="medicalai/ClinicalBert", tokenizer=tokenizer)

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


def substitute_masked_words(sentence):
    
    words = sentence.split()

    global j
    j = j + 1
    print(j)

    mask_indices = [i for i, x in enumerate(sentence.split()) if x == "[MASK]"]
    try:
        predictions = classifier(sentence)
    except:
        return "NULL"

    mask_indices = [i for i, x in enumerate(words) if x == "[MASK]"]
    
    for i in range(len(mask_indices)):
        words[mask_indices[i]] = predictions[i][0]['token_str']
    return " ".join(words)
    tokenized_sentence = tokenizer.tokenize(sentence)
    for i, token in enumerate(tokenized_sentence):
        if token == "[MASK]":
            predictions = classifier(sentence)
            try:
                chunk = predictions[0]['token_str']
            except:
                chunk = predictions[0][0]['token_str']
            predicted_word = chunk
            tokenized_sentence[i] = predicted_word
    return tokenizer.convert_tokens_to_string(tokenized_sentence)



#print(substitute_masked_words(dataset['masked'].values[0]))
dataset['predicted'] = dataset['masked'].apply(substitute_masked_words)


dataset.to_csv(f'medicalData\Medical-Abstracts-TC-Corpus\medical_tc_test_masked_medical_4.csv', index=False)