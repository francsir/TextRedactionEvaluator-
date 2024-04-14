from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import pandas as pd

import pickle

train_dataset = pd.read_csv('medical_tc_train.csv')

##drop anything with label 5
train_dataset = train_dataset[train_dataset['condition_label'] != 5]

#test_dataset = pd.read_csv('medicalData5.csv')
#test_dataset = pd.read_csv('medical_tc_test.csv')
test_dataset = pd.read_csv('medical_tc_test_masked_medical_3.csv')
test_dataset = test_dataset[test_dataset['condition_label'] != 5]
#DROP ALL NULL VALUESin predicted column
#test_dataset = test_dataset.dropna(subset=['predicted'])
test_dataset = test_dataset[:100]


X_train = train_dataset['medical_abstract']
y_train = train_dataset['condition_label']

X_test = test_dataset['masked']
y_test = test_dataset['condition_label']

tfidf_vectoizer = TfidfVectorizer(max_features=5000)
X_train = tfidf_vectoizer.fit_transform(X_train)
X_test = tfidf_vectoizer.transform(X_test)

nb_classifier = MultinomialNB(alpha=0.1)

nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report")
print(classification_report(y_test, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

with open('naive_bayes_model.pkl', 'wb') as f:
    pickle.dump((nb_classifier, tfidf_vectoizer), f)
