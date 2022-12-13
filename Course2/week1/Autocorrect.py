import re
from collections import Counter
import numpy as np
import pandas as pd
import os
import w1_unittest


# UNQ_C1 GRADED FUNCTION: process_data
def process_data(file_name):
    """
    Input:
        A file_name which is found in your current directory. You just have to read it in.
    Output:
        words: a list containing all the words in the corpus (text file you read) in lower case.
    """
    words = []  # return this variable correctly

    ### START CODE HERE ###

    # Open the file, read its contents into a string variable
    with open(file_name) as f:
        text=f.read()
    # convert all letters to lower case
    lowercase=text.lower()
    # Convert every word to lower case and return them in a list.
    words=re.findall('\w+',lowercase)
    ### END CODE HERE ###

    return words


#DO NOT MODIFY THIS CELL
word_l = process_data('./data/shakespeare.txt')
vocab = set(word_l)  # this will be your new vocabulary
print(f"The first ten words in the text are: \n{word_l[0:10]}")
print(f"There are {len(vocab)} unique words in the vocabulary.")