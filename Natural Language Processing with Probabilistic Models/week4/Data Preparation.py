import re
import nltk

nltk.download('punkt')

import emoji
import numpy as np
from nltk.tokenize import word_tokenize
from utils2 import get_dict



# Define a corpus
corpus = 'Who ❤️ "word embeddings" in 2020? I do!!!'

# print original corpus
print(f'Corpus: {corpus}')
# substitution
data=re.sub(r'[,!?;-]+','.',corpus)
# print cleaned corpus
print(f'Cleaned Corpus: {data}')
# print cleand corpus
print(f'Initial string: {data}')
# tokenize the cleaned corpus
data=nltk.word_tokenize(data)
# print tokenize string
print(f'After tokenization: {data}')