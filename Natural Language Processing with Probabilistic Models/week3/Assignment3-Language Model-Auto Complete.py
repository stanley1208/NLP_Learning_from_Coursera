import math
import random
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.data.path.append('.')

with open('./data/en_US.twitter.txt','r') as f:
    data=f.read()
print("Data type:", type(data))
print("Number of letters:", len(data))
print("First 300 letters of the data")
print("-------")
print(data[0:300])
print("-------")

print("Last 300 letters of the data")
print("-------")
print(data[-300:])
print("-------")


# UNIT TEST COMMENT: Candidate for Table Driven Tests
### UNQ_C1 GRADED_FUNCTION: split_to_sentences ###
def split_to_sentences(data):
    """
    Split data by linebreak "\n"

    Args:
        data: str

    Returns:
        A list of sentences
    """
    ### START CODE HERE ###
    sentences = list(data.split('\n'))
    ### END CODE HERE ###

    # Additional clearning (This part is already implemented)
    # - Remove leading and trailing spaces from each sentence
    # - Drop sentences if they are empty strings.
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]

    return sentences


# test your code
x = """
I have a pen.\nI have an apple. \nAh\nApple pen.\n
"""
print(x)

print(split_to_sentences(x))