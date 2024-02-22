from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer, util
from datasets import load_from_disk
import matplotlib as plt



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
for i in range(len(dataset_loaded['text'])):
    

    masked_word = dataset_loaded['label'][i]
    masked_embedding = embeding_model.encode(masked_word, convert_to_tensor=True)
    text = dataset_loaded['text'][i]

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
        j = j + 1
        
        points = len(preds) - preds.index(pred)
        redaction_score += (points * cosine_similarity)
    print(redaction_score)

print(correct_guess)




    