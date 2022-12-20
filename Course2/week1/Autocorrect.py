import re
from collections import Counter
import numpy as np
import pandas as pd
import os



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


# UNIT TEST COMMENT: Candidate for Table Driven Tests
# UNQ_C2 GRADED FUNCTION: get_count
def get_count(word_l):
    '''
    Input:
        word_l: a set of words representing the corpus.
    Output:
        word_count_dict: The wordcount dictionary where key is the word and value is its frequency.
    '''

    word_count_dict = {}  # fill this with word counts
    ### START CODE HERE
    word_count_dict=Counter()
    for word in word_l:
        word_count_dict[word]+=1
    ### END CODE HERE ###
    return word_count_dict


#DO NOT MODIFY THIS CELL
word_count_dict = get_count(word_l)
print(f"There are {len(word_count_dict)} key values pairs")
print(f"The count for the word 'thee' is {word_count_dict.get('thee',0)}")


# UNQ_C3 GRADED FUNCTION: get_probs
def get_probs(word_count_dict):
    '''
    Input:
        word_count_dict: The wordcount dictionary where key is the word and value is its frequency.
    Output:
        probs: A dictionary where keys are the words and the values are the probability that a word will occur.
    '''
    probs = {}  # return this variable correctly

    ### START CODE HERE ###
    total=sum(word_count_dict.values())
    # get the total count of words for all words in the dictionary
    for word,count in word_count_dict.items():
        probs[word]=count/total

    ### END CODE HERE ###
    return probs


#DO NOT MODIFY THIS CELL
probs = get_probs(word_count_dict)
print(f"Length of probs is {len(probs)}")
print(f"P('thee') is {probs['thee']:.4f}")


# UNIT TEST COMMENT: Candidate for Table Driven Tests
# UNQ_C4 GRADED FUNCTION: deletes
def delete_letter(word, verbose=False):
    '''
    Input:
        word: the string/word for which you will generate all possible words
                in the vocabulary which have 1 missing character
    Output:
        delete_l: a list of all possible strings obtained by deleting 1 character from word
    '''

    delete_l = []
    split_l = []

    ### START CODE HERE ###
    split_l=[(word[:i],word[i:]) for i in range(len(word))]
    delete_l=[L+R[1:] for L,R in split_l if R]
    ### END CODE HERE ###

    if verbose: print(f"input word {word}, \nsplit_l = {split_l}, \ndelete_l = {delete_l}")

    return delete_l


delete_word_l = delete_letter(word="cans",
                        verbose=True)
print(delete_word_l)

# test # 2
print(f"Number of outputs of delete_letter('at') is {len(delete_letter('at'))}")