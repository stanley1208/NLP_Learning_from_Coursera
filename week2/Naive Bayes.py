from utils import process_tweet,lookup
import pdb
from nltk.corpus import stopwords,twitter_samples
import numpy as np
import pandas as pd
import nltk
import string
from nltk.tokenize import TweetTokenizer
from os import getcwd
import w2_unittest

nltk.download('twitter_samples')
nltk.download('stopwords')

filePath=f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)

# get the sets of positive and negative tweets
all_positive_tweets=twitter_samples.strings('positive_tweets.json')
all_negative_tweets=twitter_samples.strings('negative_tweets.json')

# split the data into two pieces, one for training and one for testing (validation set)
test_pos=all_positive_tweets[4000:]
train_pos=all_positive_tweets[:4000]
test_neg=all_negative_tweets[4000:]
train_neg=all_negative_tweets[:4000]

train_x=train_pos+train_neg
test_x=test_pos+test_neg

# avoid assumptions about the length of all_positive_tweets
train_y=np.append(np.ones(len(train_pos)),np.zeros(len(train_neg)))
test_y=np.append(np.ones(len(test_pos)),np.zeros(len(test_neg)))

custom_tweet = "RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning http://chapagain.com.np"

# print cleaned tweet
# print(process_tweet(custom_tweet))

# UNQ_C1 GRADED FUNCTION: count_tweets

def count_tweets(result, tweets, ys):
    '''
    Input:
        result: a dictionary that will be used to map each pair to its frequency
        tweets: a list of tweets
        ys: a list corresponding to the sentiment of each tweet (either 0 or 1)
    Output:
        result: a dictionary mapping each pair to its frequency
    '''
    ### START CODE HERE ###
    for y, tweet in zip(ys, tweets):
        for word in process_tweet(tweet):
            # define the key, which is the word and label tuple
            pair = (word,y)

            # if the key exists in the dictionary, increment the count
            if pair in result:
                result[pair] += 1

            # else, if the key is new, add it to the dictionary and set the count to 1
            else:
                result[pair] = 1
    ### END CODE HERE ###


    return result

# Testing your function
result = {}
tweets = ['i am happy', 'i am tricked', 'i am sad', 'i am tired', 'i am tired']
ys = [1, 0, 0, 0, 0]
print(count_tweets(result, tweets, ys))

# Test your function
w2_unittest.test_count_tweets(count_tweets)

# Build the freqs dictionary for later uses
freqs = count_tweets({}, train_x, train_y)


# UNQ_C2 GRADED FUNCTION: train_naive_bayes

def train_naive_bayes(freqs, train_x, train_y):
    '''
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels correponding to the tweets (0,1)
    Output:
        logprior: the log prior. (equation 3 above)
        loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
    '''
    loglikelihood = {}
    logprior = 0

    ### START CODE HERE ###

    # calculate V, the number of unique words in the vocabulary
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    # calculate N_pos, N_neg, V_pos, V_neg
    N_pos = N_neg = 0
    for pair in freqs.keys():
        # if the label is positive (greater than zero)
        if pair[1] > 0:

            # Increment the number of positive words by the count for this (word, label) pair
            N_pos += freqs[pair]

        # else, the label is negative
        else:

            # increment the number of negative words by the count for this (word,label) pair
            N_neg += freqs[pair]

    # Calculate D, the number of documents
    D = len(train_y)

    # Calculate D_pos, the number of positive documents
    D_pos = np.sum(train_y)

    # Calculate D_neg, the number of negative documents
    D_neg = D-D_pos

    # Calculate logprior
    logprior = np.log(D_pos/D_neg)

    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos = lookup(freqs,word,1)
        freq_neg = lookup(freqs,word,0)

        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos+1)/(N_pos+V)
        p_w_neg = (freq_neg+1)/(N_neg+V)

        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos/p_w_neg)

    ### END CODE HERE ###

    return logprior, loglikelihood


# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
print(logprior)
print(len(loglikelihood))

# Test your function
# w2_unittest.test_train_naive_bayes(train_naive_bayes, freqs, train_x, train_y)