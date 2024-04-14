from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer, util
from datasets import load_from_disk
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import json
import numpy as np


def save_dicts(corr_guess, true_counts, pred_counts):
    data = {
        "correct_guest_distribution": corr_guess,
        "pos_true_counts": true_counts,
        "pos_corr_pred_counts":pred_counts,
    }
    with open("prediction_res.json", 'w') as file:
        json.dump(data,file, indent=4)

def insertPos(pos, counts):
    if pos in counts:
        counts[pos] += 1
    else:
        counts[pos] = 1
    
def plot_counts(tr_pos, prd_pos):
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

def eval(pred_word, true_word, embeding_model):

    masked_embedding = embeding_model.encode(true_word, convert_to_tensor=True)
    prediction_embedding = embeding_model.encode(pred_word['token'], convert_to_tensor= True)
    cosine_similarity = util.pytorch_cos_sim(masked_embedding, prediction_embedding)

    print(f"{pred_word['token_str']}: {pred_word['score']:.2f}")
    print(f"Cosine Similarity: {cosine_similarity.item():.2f}")



## Load Models:
    

output_dir = "model/imdb-finetuned-distilbert"
embeding_model = SentenceTransformer('all-MiniLM-L6-v2')
model = AutoModelForMaskedLM.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

mask_pipeline = pipeline("fill-mask", model=model, tokenizer=tokenizer)
##
#### Load Data:

dataset_loaded = load_from_disk('./datasets/imdb')
print(dataset_loaded)
correct_guess = {
        "1": 0, 
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
    }
pos_true_counts = {}
pos_cor_pred_counts = {}

for i in range(len(dataset_loaded['text'])):
    

    masked_word = dataset_loaded['label'][i]
    masked_embedding = embeding_model.encode(masked_word, convert_to_tensor=True)
    text = dataset_loaded['text'][i]


    ## POS Classify
    pos_tags = pos_tag([dataset_loaded['label'][i]])
    _, pos = pos_tags[0]
    insertPos(pos, pos_true_counts)
    ## Make Predictions:

    preds = mask_pipeline(text)
    #print(f'>>>{i}----------------------------------------')
    redaction_score = 0
    ## Calculate Similarity:
    
    j = 1
    for pred in preds:
        prediction_embedding = embeding_model.encode(pred['token_str'], convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(masked_embedding, prediction_embedding)

        #print(f"{pred['token_str']}: {pred['score']:.2f}")
        #print(f"Cosine Similarity: {cosine_similarity.item():.2f}")

        if(cosine_similarity.item() == 1):
            correct_guess[str(j)] += 1
            insertPos(pos, pos_cor_pred_counts)
        j = j + 1
        
        points = len(preds) - preds.index(pred)
        redaction_score += (points * cosine_similarity)
    print(redaction_score)

print(f"correct guess distribution: {correct_guess}")
print(f"pos true counts: {pos_true_counts}")
print(f"pos corr pred counts: {pos_cor_pred_counts}")
save_dicts(corr_guess=correct_guess, true_counts=pos_true_counts, pred_counts=pos_cor_pred_counts)
plot_counts(pos_true_counts, pos_cor_pred_counts)

    