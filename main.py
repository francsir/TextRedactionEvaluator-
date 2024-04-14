#from maskData import __main__ as maskData
from predictionEvaluator import __main__ as predictionEvaluator
#from read_npy import __main__ as read_npy
import os

MAX_MASKED_WORDS = 5
DATASET_SIZE = 2

def __main__():

    ## Mask the words in the dataset
    #for i in range(1, MAX_MASKED_WORDS):
    #    maskData(mask_count = i, size = DATASET_SIZE)

    #print("Masking Done")

     
    for folder_name in os.listdir("datasets"):
        predictionEvaluator(folder_name)
    
    print("Prediction Evaluation Done")

    #read_npy()
    print("Read NPY Done")

    








__main__()
