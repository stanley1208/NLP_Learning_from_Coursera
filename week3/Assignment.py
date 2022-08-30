import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import w3_unittest
from utils import get_vectors


data=pd.read_csv('./data/capitals.txt',delimiter=' ')
data.columns=['city1','country1','city2','country2']

# print first five elements in the DataFrame
# print(data.head(5))


word_embeddings=pickle.load(open("./data/word_embeddings_subset.p","rb"))
print(len(word_embeddings))
print("dimension:{}".format(word_embeddings['Spain'].shape[0]))

# UNQ_C1 GRADED FUNCTION: cosine_similarity

def cosine_similarity(A, B):
    '''
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    '''

    ### START CODE HERE ###
    dot = np.dot(A,B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    cos = dot/(norma*normb)

    ### END CODE HERE ###
    return cos

king=word_embeddings['king']
queen=word_embeddings['queen']

print(cosine_similarity(king,queen))

# UNQ_C2 GRADED FUNCTION: euclidean

def euclidean(A, B):
    """
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        d: numerical number representing the Euclidean distance between A and B.
    """

    ### START CODE HERE ###

    # euclidean distance
    d = np.linalg.norm(A-B)

    ### END CODE HERE ###

    return d

