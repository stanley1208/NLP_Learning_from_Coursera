import nltk                                     # Python library for NLP
from nltk.corpus import twitter_samples         # sample Twitter dataset from NLTK
import matplotlib.pyplot as plt                 # library for visualization
import random                                   # pseudo-random number generator
import re                                       # library for regular expression operations
import string                                   # for string operations
from nltk.corpus import stopwords               # module for stop words that come with NLTK
from nltk.stem import PorterStemmer             # module for stemming
from nltk.tokenize import TweetTokenizer        # module for tokenizing strings
from utils import process_tweet                 # Import the process_tweet function
from utils import build_freqs


# downloads sample twitter dataset.
nltk.download('twitter_samples')

# select the set of positive and negative tweets
all_positive_tweets=twitter_samples.strings('positive_tweets.json')
all_negative_tweets=twitter_samples.strings('negative_tweets.json')

# Our selected sample. Complex enough to exemplify each step
tweet=all_positive_tweets[2277]


print()
print('\033[92m')
print(tweet)
print('\033[94m')

# call the imported function
tweets_stem=process_tweet(tweet)  # Preprocess a given tweet

print('preprocessed tweet:')
print(tweets_stem)
