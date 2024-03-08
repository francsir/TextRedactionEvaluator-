from sympy import evaluate
from transformers import pipeline,  AutoTokenizer, AutoModelForMaskedLM
from nltk import pos_tag
from datasets import load_from_disk
from sentence_transformers import util
import gensim.downloader as api
import matplotlib.pyplot as plt
import numpy as np
import json
import csv



class RedactionEvaluator:
    
    def __init__(self, maskedLM, tokenizer, word2vec):
        self.maskedLm = maskedLM
        self.tokenizer = tokenizer
        
        if(word2vec):
            self.word2vec = api.load('word2vec-google-news-300')

        self.mask_pipeline = pipeline("fill-mask", model=self.maskedLm, tokenizer=self.tokenizer)

    ## Initialize the dictionaries
    def init_dicts(self, size):
        self.correct_guess = {str(i): 0 for i in range(1, size + 1)}
        self.pos_true_counts = {}
        self.pos_pred_counts = {}
        self.pos_distances = {}
        
        return self.correct_guess, self.pos_true_counts, self.pos_pred_counts

    ## Get the part of speech tag for a word
    def get_pos_tag(self, word):
        pos_tags = pos_tag(word)
        _, pos =pos_tags[0]
        return pos
    
    ## Get the predictions for the masked word
    def make_preds(self, text):
        return self.mask_pipeline(text)
    
    ## Get the cosine similarity between two words
    def cosine_sim(self, word1, word2):
        return util.pytorch_cos_sim(word1, word2)
    
    ## Save the dictionaries to a file
    def save_dicts(self):
        data = {
            "correct_guest_distribution": self.correct_guess,
            "pos_true_counts": self.pos_true_counts,
            "pos_corr_pred_counts": self.pos_pred_counts,
        }
        np.save("pos_distance.npy", self.pos_distances)
        
        with open("prediction_res.json", 'w') as file:
            json.dump(data, file, indent=4)

    ## Insert the part of speech tag into the dictionary/update its count
    def insert_pos(self, pos, counts):
        if pos in counts:
            counts[pos] += 1
        else:
            counts[pos] = 1

    ## Insert the part of speech tag and its distance into the dictionary
    def insert_pos_distance(self, pos,distance):
        if pos in self.pos_distances:
            self.pos_distances[pos].append(distance)
        else:
            self.pos_distances[pos] = [distance]
    
    ## Plot the part of speech tag counts
    def plot_pos_counts(self):
        tr_pos = self.pos_true_counts
        prd_pos = self.pos_pred_counts
        all_pos = set(list(tr_pos.keys()) + list(prd_pos.keys()))

        pos_types = sorted(list(all_pos))
        tr_counts = [tr_pos.get(pos, 0) for pos in pos_types]
        corr_counts = [prd_pos.get(pos, 0) for pos in pos_types]

        bar_width = 0.35
        index = np.arange(len(pos_types))
        opacity = 0.8

        plt.bar(index, tr_counts, bar_width, alpha=opacity, color='b', label='True Counts')
        plt.bar(index + bar_width, corr_counts, bar_width, alpha=opacity, color='g', label='Correct Predictions')

        plt.xlabel('POS Types')
        plt.ylabel('Counts')
        plt.title('True POS Type Counts vs Correct POS Predictions')
        plt.xticks(index + bar_width / 2, pos_types)
        plt.legend()

        plt.tight_layout()
        plt.show()

def save_csv():
    csv_file_path = "predictions.csv"

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        writer.writerow(["Sentence", "True Words", "Predicted Words", "Scores"])

        for sentence, true_word, pred_word, score in zip(sentences, true_words, pred_words, scores):
            writer.writerow([sentence,', '.join(true_word), ', '.join(pred_word),', '.join(map(str, score))])



output_dir = "model/imdb-finetuned-distilbert"
maskedLM = AutoModelForMaskedLM.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir) 
evaluator = RedactionEvaluator(maskedLM, tokenizer, True)

dataset = load_from_disk('./datasets/imdb')
evaluator.init_dicts(5)

sentences = []
pred_words = []
true_words = []
scores = []
    
for i in range(len(dataset['text'])):
    
    ## Get the true value of the masked word
    masked_word = dataset['label'][i].lower()
    if masked_word not in evaluator.word2vec:
        print(f'{masked_word}: Masked Word not in word2vec, skipping')
        continue
    
    masked_embeding = evaluator.word2vec[masked_word]
    true_words.append(masked_word)
    
    ## Get the words Part of speech label
    pos = evaluator.get_pos_tag([masked_word])
    evaluator.insert_pos(pos, evaluator.pos_true_counts)
    
    ## load the masked sentence and get predictions for the mask
    text = dataset['text'][i]
    sentences.append(text)

    preds = evaluator.make_preds(text)
    
    j = 1

    if preds is None:
        break

    temp_preds = []
    temp_scores = []
    for pred in preds:
        if pred['token_str'] not in evaluator.word2vec:
            print(f'{pred["token_str"]}: word not in word2vec vocab')
            continue

        temp_preds.append(pred['token_str'])

        pred_embeding = evaluator.word2vec[pred['token_str']]
        mean_squared_distance = np.mean((masked_embeding - pred_embeding) ** 2)

        if(mean_squared_distance == 0):
            evaluator.correct_guess[str(j)] += 1
            evaluator.insert_pos(pos, evaluator.pos_pred_counts)
            
        j = j + 1
        
        masked_pos = evaluator.get_pos_tag([pred['token_str']])
        #if(masked_pos == pos):
        #    
        evaluator.insert_pos_distance(pos, mean_squared_distance)
        temp_scores.append(mean_squared_distance)
    pred_words.append(temp_preds)
    scores.append(temp_scores)

save_csv()
#print(evaluator.pos_distances)
evaluator.save_dicts()
print(evaluator.pos_distances)
        
        
    
