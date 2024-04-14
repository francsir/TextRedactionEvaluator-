import pandas as pd

train_dataset = pd.read_csv('medicalData\Medical-Abstracts-TC-Corpus\medical_tc_train.csv')

X_train = train_dataset['medical_abstract']
y_train = train_dataset['condition_label']

print(X_train)
print(y_train)