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


# UNIT TEST COMMENT: Candidate for Table Driven Tests
# UNQ_C5 GRADED FUNCTION: switches
def switch_letter(word, verbose=False):
    '''
    Input:
        word: input string
     Output:
        switches: a list of all possible strings with one adjacent charater switched
    '''

    switch_l = []
    split_l = []

    ### START CODE HERE ###
    split_l = [(word[:i], word[i:]) for i in range(len(word))]
    switch_l = [L + R[1]+R[0]+R[2:] for L, R in split_l if len(R)>=2]
    ### END CODE HERE ###

    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nswitch_l = {switch_l}")

    return switch_l


switch_word_l = switch_letter(word="eta",
                         verbose=True)

print(switch_word_l)

# test # 2
print(f"Number of outputs of switch_letter('at') is {len(switch_letter('at'))}")


# UNIT TEST COMMENT: Candidate for Table Driven Tests
# UNQ_C6 GRADED FUNCTION: replaces
def replace_letter(word, verbose=False):
    '''
    Input:
        word: the input string/word
    Output:
        replaces: a list of all possible strings where we replaced one letter from the original word.
    '''

    letters = 'abcdefghijklmnopqrstuvwxyz'

    replace_l = []
    split_l = []

    ### START CODE HERE ###
    split_l = [(word[:i], word[i:]) for i in range(len(word))]
    replace_l = [a+i+(b[1:] if len(b)>1 else "") for a, b in split_l if b for i in letters]
    replace_set=set(replace_l)
    replace_set.discard(word)
    ### END CODE HERE ###

    # turn the set back into a list and sort it, for easier viewing
    replace_l = sorted(list(replace_set))

    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nreplace_l {replace_l}")

    return replace_l

replace_l = replace_letter(word='can',
                              verbose=True)

print(replace_l)
# test # 2
print(f"Number of outputs of replace_letter('at') is {len(replace_letter('at'))}")


# UNIT TEST COMMENT: Candidate for Table Driven Tests
# UNQ_C7 GRADED FUNCTION: inserts
def insert_letter(word, verbose=False):
    '''
    Input:
        word: the input string/word
    Output:
        inserts: a set of all possible strings with one new letter inserted at every offset
    '''
    letters = 'abcdefghijklmnopqrstuvwxyz'
    insert_l = []
    split_l = []

    ### START CODE HERE ###
    split_l = [(word[:i], word[i:]) for i in range(len(word)+1)]
    insert_l = [a + i + b for a, b in split_l for i in letters]
    ### END CODE HERE ###

    if verbose: print(f"Input word {word} \nsplit_l = {split_l} \ninsert_l = {insert_l}")

    return insert_l

insert_l = insert_letter('at', True)
print(insert_l)
print(f"Number of strings output by insert_letter('at') is {len(insert_l)}")


# UNIT TEST COMMENT: Candidate for Table Driven Tests
# UNQ_C8 GRADED FUNCTION: edit_one_letter
def edit_one_letter(word, allow_switches=True):
    """
    Input:
        word: the string/word for which we will generate all possible wordsthat are one edit away.
    Output:
        edit_one_set: a set of words with one possible edit. Please return a set. and not a list.
    """

    edit_one_set = set()

    ### START CODE HERE ###
    edit_one_set=delete_letter(word)
    if allow_switches:
        edit_one_set+=switch_letter(word)
    edit_one_set+=replace_letter(word)
    edit_one_set+=insert_letter(word)
    ### END CODE HERE ###

    # return this as a set and not a list
    return set(edit_one_set)


tmp_word = "at"
tmp_edit_one_set = edit_one_letter(tmp_word)
# turn this into a list to sort it, in order to view it
tmp_edit_one_l = sorted(list(tmp_edit_one_set))

print(f"input word {tmp_word} \nedit_one_l \n{tmp_edit_one_l}\n")
print(f"The type of the returned object should be a set {type(tmp_edit_one_set)}")
print(f"Number of outputs from edit_one_letter('at') is {len(edit_one_letter('at'))}")


# UNIT TEST COMMENT: Candidate for Table Driven Tests
# UNQ_C9 GRADED FUNCTION: edit_two_letters
def edit_two_letters(word, allow_switches=True):
    '''
    Input:
        word: the input string/word
    Output:
        edit_two_set: a set of strings with all possible two edits
    '''

    edit_two_set = set()

    ### START CODE HERE ###
    edit_1=edit_one_letter(word,allow_switches=allow_switches)
    for i in edit_1:
        edit_2=edit_one_letter(i,allow_switches=allow_switches)
        edit_two_set.update(edit_2)
    ### END CODE HERE ###

    # return this as a set instead of a list
    return set(edit_two_set)


tmp_edit_two_set = edit_two_letters("a")
tmp_edit_two_l = sorted(list(tmp_edit_two_set))
print(f"Number of strings with edit distance of two: {len(tmp_edit_two_l)}")
print(f"First 10 strings {tmp_edit_two_l[:10]}")
print(f"Last 10 strings {tmp_edit_two_l[-10:]}")
print(f"The data type of the returned object should be a set {type(tmp_edit_two_set)}")
print(f"Number of strings that are 2 edit distances from 'at' is {len(edit_two_letters('at'))}")