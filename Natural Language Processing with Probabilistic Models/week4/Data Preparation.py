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
corpus='I am happ because I am learning.'
print(f'Corpus: {corpus}')
data=tokenize(corpus)
print(f'After tokenize: {data}')

corpus='I like her, I hope she can like me as well and talk to me!'
words=tokenize(corpus)
print(data)

# Print 'context_words' and 'center_word' for the new corpus with a 'context half-size' of 2
for x, y in get_windows(['i', 'am', 'happy', 'because', 'i', 'am', 'learning'], 2):
    print(f'{x}\t{y}')

# Print 'context_words' and 'center_word' for the new corpus with a 'context half-size' of 1
for x, y in get_windows(tokenize("I like her so much but she do not like me as I do."), 1):
    print(f'{x}\t{y}')


# Get 'word2Ind' and 'Ind2word' dictionaries for the tokenized corpus
word2Ind, Ind2word = get_dict(words)

print(word2Ind)

print("Index of the word 'i': ",word2Ind['i'])
