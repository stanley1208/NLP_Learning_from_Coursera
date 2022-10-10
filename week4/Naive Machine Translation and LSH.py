import pdb
import pickle
import string
import time
import nltk
import numpy as np
from nltk.corpus import stopwords, twitter_samples
from utils import (cosine_similarity, get_dict,process_tweet)
from os import getcwd
import w4_unittest


nltk.download('stopwords')
nltk.download('twitter_samples')

# add folder, tmp2, from our local workspace containing pre-downloaded corpora files to nltk's data path
filePath=f"{getcwd()}/tmp2/"
nltk.data.path.append(filePath)

en_embeddings_subset=pickle.load(open("./data/en_embeddings.p","rb"))
fr_embeddings_subset=pickle.load(open("./data/fr_embeddings.p","rb"))


# loading the english to french dictionaries
en_fr_train=get_dict('./data/en-fr.train.txt')
print('The length of the English to French training dictionary is',len(en_fr_train))
en_fr_test=get_dict('./data/en-fr.test.txt')
print('The length of the English to French test dictionary is',len(en_fr_test))


# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def get_matrices(en_fr, french_vecs, english_vecs):
    """
    Input:
        en_fr: English to French dictionary
        french_vecs: French words to their corresponding word embeddings.
        english_vecs: English words to their corresponding word embeddings.
    Output:
        X: a matrix where the columns are the English embeddings.
        Y: a matrix where the columns correspong to the French embeddings.
        R: the projection matrix that minimizes the F norm ||X R -Y||^2.
    """

    ### START CODE HERE ###

    # X_l and Y_l are lists of the english and french word embeddings
    X_l = list()
    Y_l = list()

    # get the english words (the keys in the dictionary) and store in a set()
    english_set = set(english_vecs.keys())

    # get the french words (keys in the dictionary) and store in a set()
    french_set = set(french_vecs.keys())

    # store the french words that are part of the english-french dictionary (these are the values of the dictionary)
    french_words = set(en_fr.values())

    # loop through all english, french word pairs in the english french dictionary
    for en_word, fr_word in en_fr.items():

        # check that the french word has an embedding and that the english word has an embedding
        if fr_word in french_set and en_word in english_set:

            # get the english embedding
            en_vec = english_vecs[en_word]

            # get the french embedding
            fr_vec = french_vecs[fr_word]

            # add the english embedding to the list
            X_l.append(en_vec)

            # add the french embedding to the list
            Y_l.append(fr_vec)

    # stack the vectors of X_l into a matrix X
    X = np.vstack(X_l)

    # stack the vectors of Y_l into a matrix Y
    Y = np.vstack(Y_l)
    ### END CODE HERE ###

    return X, Y



# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# You do not have to input any code in this cell, but it is relevant to grading, so please do not change anything

# getting the training set:
X_train, Y_train = get_matrices(en_fr_train, fr_embeddings_subset, en_embeddings_subset)