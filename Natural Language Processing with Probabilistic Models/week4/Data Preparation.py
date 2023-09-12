import re
import nltk

nltk.download('punkt')

import emoji
import numpy as np
from nltk.tokenize import word_tokenize
from utils2 import get_dict



# Define a corpus
corpus = 'Who ❤️ "word embeddings" in 2020? I do!!!'
