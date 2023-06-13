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


# test your function
x = "Sky is blue.\nLeaves are green\nRoses are red."
print(get_tokenized_data(x))




tokenized_data=get_tokenized_data(data)
random.seed(87)
random.shuffle(tokenized_data)

train_size=int(len(tokenized_data)*0.8)
train_data=tokenized_data[0:train_size]
test_data=tokenized_data[train_size:]


print("{} data are split into {} train and {} test set".format(len(tokenized_data),len(train_data),len(test_data)))

print("First training sample:",train_data[0])
print("First test sample:",test_data[0])


# UNIT TEST COMMENT: Candidate for Table Driven Tests
### UNQ_C4 GRADED_FUNCTION: count_words ###
def count_words(tokenized_sentences):
    """
    Count the number of word appearence in the tokenized sentences

    Args:
        tokenized_sentences: List of lists of strings

    Returns:
        dict that maps word (str) to the frequency (int)
    """

    word_counts = {}
    ### START CODE HERE ###

    # Loop through each sentence
    for sentence in tokenized_sentences:  # complete this line

        # Go through each token in the sentence
        for token in sentence:  # complete this line

            # If the token is not in the dictionary yet, set the count to 1
            if token not in word_counts:  # complete this line with the proper condition
                word_counts[token] = 1

            # If the token is already in the dictionary, increment the count by 1
            else:
                word_counts[token] += 1

    ### END CODE HERE ###

    return word_counts



# test your code
tokenized_sentences = [['sky', 'is', 'blue', '.'],
                       ['leaves', 'are', 'green', '.'],
                       ['roses', 'are', 'red', '.']]
print(count_words(tokenized_sentences))