from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer, util

## Load Models:
    
output_dir = "model/imdb-finetuned-distilbert"
embeding_model = SentenceTransformer('all-MiniLM-L6-v2')
model = AutoModelForMaskedLM.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

mask_pipeline = pipeline("fill-mask", model=model, tokenizer=tokenizer)

## Load Data:

masked_word = 'horror'
masked_embedding = embeding_model.encode(masked_word, convert_to_tensor=True)
text = "He walked into the basement with the [MASK] movie from the night before playing in his head."

## Make Predictions:

preds = mask_pipeline(text)

## Calculate Similarity:
for pred in preds:
    prediction_embedding = embeding_model.encode(pred['token_str'], convert_to_tensor=True)
    cosine_similarity = util.pytorch_cos_sim(masked_embedding, prediction_embedding)

    print(f"{pred['token_str']}: {pred['score']:.2f}")
    print(f"Cosine Similarity: {cosine_similarity.item():.2f}")



    