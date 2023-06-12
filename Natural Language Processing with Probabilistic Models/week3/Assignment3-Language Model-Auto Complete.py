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


# UNIT TEST COMMENT: Candidate for Table Driven Tests
### UNQ_C2 GRADED_FUNCTION: tokenize_sentences ###
def tokenize_sentences(sentences):
    """
    Tokenize sentences into tokens (words)

    Args:
        sentences: List of strings

    Returns:
        List of lists of tokens
    """

    # Initialize the list of lists of tokenized sentences
    tokenized_sentences = []
    ### START CODE HERE ###

    # Go through each sentence
    for sentence in sentences:  # complete this line

        # Convert to lowercase letters
        sentence = sentence.lower()

        # Convert into a list of words
        tokenized = nltk.word_tokenize(sentence)

        # append the list of words to the list of lists
        tokenized_sentences.append(tokenized)

    ### END CODE HERE ###

    return tokenized_sentences


# test your code
sentences = ["Sky is blue.", "Leaves are green.", "Roses are red."]
print(tokenize_sentences(sentences))



# UNIT TEST COMMENT: Candidate for Table Driven Tests
### UNQ_C3 GRADED_FUNCTION: get_tokenized_data ###
def get_tokenized_data(data):
    """
    Make a list of tokenized sentences

    Args:
        data: String

    Returns:
        List of lists of tokens
    """
    ### START CODE HERE ###

    # Get the sentences by splitting up the data
    sentences = split_to_sentences(data)

    # Get the list of lists of tokens by tokenizing the sentences
    tokenized_sentences = tokenize_sentences(sentences)

    ### END CODE HERE ###

    return tokenized_sentences