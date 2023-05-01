import numpy as np
import pandas as pd
from collections import defaultdict


def single_pass_trigram_count_matrix(corpus):
    """
    Creates the trigram count matrix from the input corpus in a single pass through the corpus.

    Args:
        corpus: Pre-processed and tokenized corpus.

    Returns:
        bigrams: list of all bigram prefixes, row index
        vocabulary: list of all found words, the column index
        count_matrix: pandas dataframe with bigram prefixes as rows,
                      vocabulary words as columns
                      and the counts of the bigram/word combinations (i.e. trigrams) as values
    """
    bigrams=[]
    vocabulary=[]
    count_matrix_dict=defaultdict(dict)

    # go through the corpus once with a sliding window
    for i in range(len(corpus)-2):
        trigram=tuple(corpus[i:i+3])

        bigram=trigram[0:-1]
        if not bigram in bigrams:
            bigrams.append(bigram)

        last_word=trigram[-1]
        if not last_word in vocabulary:
            vocabulary.append(last_word)

        if (bigram,last_word) not in count_matrix_dict:
            count_matrix_dict[bigram,last_word]=0

        count_matrix_dict[bigram, last_word]+=1

        # convert the count_matrix to np.array to fill in the blanks
    count_matrix = np.zeros((len(bigrams), len(vocabulary)))
    for trigram_key, trigram_count in count_matrix_dict.items():
        count_matrix[bigrams.index(trigram_key[0]),vocabulary.index(trigram_key[1])]= trigram_count

    count_matrix=pd.DataFrame(count_matrix,index=bigrams,columns=vocabulary)

    return bigrams,vocabulary,count_matrix


corpus=['i','am','happy','because','i','am','learning','.']

bigrams,vocabulary,count_matrix=single_pass_trigram_count_matrix(corpus)

print(count_matrix)

print("-----------------------------------")

# manipulate n_gram count dictionary
n_gram_counts={
    ('i','am','happy'):2,
    ('am','happy','because'):1
}

# get count for an n-gram tuple
print(f"count of n-gram {('i','am','happy')}:{n_gram_counts[('i','am','happy')]}")

# check if n-gram is present in the dictionary
if ('i','am','learning') in n_gram_counts:
    print(f"n-gram {('i','am','learning')} found")
else:
    print(f"n-gram {('i', 'am', 'learning')} missing")

# update the count in the word count dictionary
n_gram_counts['i','am','learning']=1
if ('i','am','learning') in n_gram_counts:
    print(f"n-gram {('i','am','learning')} found")
else:
    print(f"n-gram {('i', 'am', 'learning')} missing")

# concatenate tuple for prefix and tuple with the last word to create the n_gram
prefix=('i','am','happy')
word='because'

# note here the syntax for creating a tuple for a single word
n_gram=prefix+(word,)
print(n_gram)


