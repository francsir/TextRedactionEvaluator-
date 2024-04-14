import pickle

import pandas as pd

dataset = pd.read_csv('medicalData\Medical-Abstracts-TC-Corpus\medical_tc_test.csv')


with open('naive_bayes_model.pkl', 'rb') as f:
    nb_classifier_loaded, tfidf_vectoizer = pickle.load(f)


sentence = [dataset['medical_abstract'][0]]

X_new = tfidf_vectoizer.transform(sentence)

predicted = nb_classifier_loaded.predict(X_new)

print(predicted)
