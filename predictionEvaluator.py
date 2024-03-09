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
import os

folder = 'Data'
global dataset_path 

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

        self.mean_squared_distances = {}
        
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
        global dataset_path
        data = {
            "correct_guest_distribution": self.correct_guess,
            "pos_true_counts": self.pos_true_counts,
            "pos_corr_pred_counts": self.pos_pred_counts,
        }
        np.save(f"{folder}/{dataset_path}/pos_distance.npy", self.pos_distances)
        np.save(f"{folder}/{dataset_path}/mean_squared_distances.npy", self.mean_squared_distances)
        
        with open(f"{folder}/{dataset_path}/prediction_res.json", 'w') as file:
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
    
    def insert_meansqr_pos(self, mean_sqr, pos):
        if pos in self.mean_squared_distances:
            self.mean_squared_distances[pos].append(mean_sqr)
        else:
            self.mean_squared_distances[pos] = [mean_sqr]
    
    ## Plot the part of speech tag counts
    def plot_pos_counts(self):
        global dataset_path
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

        plt.savefig(f"{folder}/{dataset_path}/pos_counts.png")
        plt.close('all')
        #plt.show()

def mean_sqr(array):
    distances = np.array(array)
    mean_squared = np.mean(distances ** 2)

    return mean_squared

def save_csv(sentences, true_words, pred_words, scores):
    global dataset_path
    csv_file_path = f"{folder}/{dataset_path}/predictions.csv"

    folder_path = f"{folder}/{dataset_path}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        writer.writerow(["Sentence", "True Words", "Predicted Words", "Scores"])

        for sentence, true_word, pred_word, score in zip(sentences, true_words, pred_words, scores):
            writer.writerow([sentence,', '.join(true_word), ', '.join(pred_word),', '.join(map(str, score))])

def __main__(ds):

    global dataset_path
    dataset_path = ds

    ## Load the model, tokenizer, evaluator and dataset
    output_dir = "model/imdb-finetuned-distilbert"
    maskedLM = AutoModelForMaskedLM.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir) 
    evaluator = RedactionEvaluator(maskedLM, tokenizer, True)
    dataset = load_from_disk(f"datasets/{dataset_path}")

    ## Initialize the dictionaries for storing pos count etc
    evaluator.init_dicts(5)

    ## Initialize the lists for storing the sentences, true words, predicted words and scores
    sentences = []
    pred_words = []
    true_words = []
    scores = []

    ## Iterate through each sentence in the dataset and get the predictions, true words and scores

    for i in range(len(dataset['text'])):

        ## Get the true value of the masked word
        true_sentence = dataset['original'][i].lower().split()
        masked_sentence = dataset['text'][i].lower().split()

        masked_indices = [i for i, (original, masked) in enumerate(zip(true_sentence, masked_sentence)) if original != masked]

        masked_word_list = [true_sentence[i] for i in masked_indices]


        true_words.append(true_sentence)
        

        ## load the masked sentence and get predictions for the mask
        ##TODO CHECK IF LOWER IMPROVES/DETERIORATES PERFORMANCE
        text = dataset['text'][i]
        sentences.append(text)

        # Make Prediciton for the masked word(s)
        preds = evaluator.make_preds(text)
        
        predictions = {}

        if preds is None:
            continue
    
        predictions = {}
        try:
            for i in range(len(preds)):
                for pred in preds[i]:
                    if str(i) not in predictions:
                        predictions[str(i)] = [pred['token_str']]
                    else:
                        predictions[str(i)].append(pred['token_str'])
        except:
            continue 

        for j in range(len(masked_word_list)):
            masked_word = masked_word_list[j]

            if masked_word not in evaluator.word2vec:
                print(f'{masked_word}: Masked Word not in word2vec, skipping')
                continue

            
            pos = evaluator.get_pos_tag([masked_word])
            evaluator.insert_pos(pos, evaluator.pos_true_counts)

            guess_iter = 1
            
            temp_preds = []
            temp_scores = []

            for pred in predictions[str(j)]:
                if pred not in evaluator.word2vec:
                    print(f'{pred}: Pred word not in word2vec vocab')
                    continue

                temp_preds.append(pred)
                distance = evaluator.word2vec.similarity(masked_word, pred)

                if(distance > 0.99 or pred == masked_word):
                    evaluator.correct_guess[str(guess_iter)] += 1
                    evaluator.insert_pos(pos, evaluator.pos_pred_counts)

                guess_iter = guess_iter + 1

                masked_pos = evaluator.get_pos_tag([pred])

                evaluator.insert_pos_distance(pos, distance)
                temp_scores.append(distance)
            
            ms = mean_sqr(temp_scores)
            evaluator.insert_meansqr_pos(ms, pos)

            pred_words.append(temp_preds)
            scores.append(temp_scores)

    save_csv(sentences, true_words, pred_words, scores)
    evaluator.plot_pos_counts()
    evaluator.save_dicts()
