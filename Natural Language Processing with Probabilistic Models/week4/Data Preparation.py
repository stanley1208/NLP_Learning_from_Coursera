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
data=tokenize(corpus)
print(data)

