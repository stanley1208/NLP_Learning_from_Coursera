import re
import nltk

nltk.download('punkt')

import emoji
import numpy as np
from nltk.tokenize import word_tokenize
from utils2 import get_dict

# substitution/tokenized/filtered
def tokenize(corpus):
    data = re.sub(r'[,!?;-]+', '.', corpus)
    data=nltk.word_tokenize(data)
    data = [ch.lower() for ch in data
            if ch.isalpha()
            or ch == '.'
            or emoji.get_emoji_regexp().search(ch)
            ]
    return data

# Define the 'get_windows' function
def get_windows(words,C):
    i=C
    while i<len(words)-C:
        center_word=words[i]
        context_word=words[(i-C):i]+words[(i+1):(i+C+1)]
        yield context_word,center_word
        i+=1



# Define a corpus
corpus = 'Who ❤️ "word embeddings" in 2020? I do!!!'

# print original corpus
print(f'Corpus: {corpus}')
# substitution
data=re.sub(r'[,!?;-]+','.',corpus)
# print cleaned corpus
print(f'Cleaned Corpus: {data}')
# print cleaned corpus
print(f'Initial string: {data}')
# tokenize the cleaned corpus
data=nltk.word_tokenize(data)
# print tokenize string
print(f'After tokenization: {data}')

# tokenized corpus
print(f'Initial list of token: {data}')
# filter tokenized corpus using list comprehension
data=[ ch.lower() for ch in data
       if ch.isalpha()
       or ch=='.'
       or emoji.get_emoji_regexp().search(ch)
    ]
# print the tokenized and filtered version of the corpus
print(f'After cleaning: {data}')

#define the corpus
corpus='I am happy because I am learning.'
print(f'Corpus: {corpus}')
data=tokenize(corpus)
print(f'After tokenize: {data}')

corpus='I am happy because I am learning'
words=tokenize(corpus)
print(data)

# Print 'context_words' and 'center_word' for the new corpus with a 'context half-size' of 2
for x, y in get_windows(['I', 'am', 'happy', 'because', 'i', 'am', 'learning'], 2):
    print(f'{x}\t{y}')

# Print 'context_words' and 'center_word' for the new corpus with a 'context half-size' of 1
for x, y in get_windows(tokenize("I am happy because I am learning."), 1):
    print(f'{x}\t{y}')


# Get 'word2Ind' and 'Ind2word' dictionaries for the tokenized corpus
word2Ind, Ind2word = get_dict(words)

print(word2Ind)

print("Index of the word 'i': ",word2Ind['i'])

print(Ind2word)

print("Which word has index 2:",Ind2word[2])

V=len(word2Ind)
print("size of the V",V)

# Save index of word 'happy' into the 'n' variable
n=word2Ind['happy']
print(n)

# Create vector with the same length as the vocabulary, filled with zeros
center_word_vector=np.zeros(V)
print(center_word_vector)

print(len(center_word_vector)==V)

# Replace element number 'n' with a 1
center_word_vector[n]=1

print(center_word_vector)


# Define the 'word_to_one_hot_vector' function that will include the steps previously seen
def word_to_one_hot_vector(word,word2Ind,V):
    one_hot_vector = np.zeros(V)
    one_hot_vector[word2Ind[word]]=1
    return one_hot_vector

print(word_to_one_hot_vector('happy',word2Ind,V))
print(word_to_one_hot_vector('i',word2Ind,V))
