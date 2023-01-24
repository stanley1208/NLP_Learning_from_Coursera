# First Steps: Working with text files
# Creating a Vocabulary
# Handling Unknown Words

import string
from collections import defaultdict

# Read lines from 'WSJ_02-21.pos' file and save them into the 'lines' variable
with open("./data/WSJ_02-21.pos",'r') as f:
    lines=f.readlines()

# Print columns for reference
print("\t\t\tWord","\tTag\n")

# Print first five lines of the dataset
for i in range(5):
    print(f'line number {i+1}:{lines[i]}')

# Print first line (unformatted)
print(lines[0])

# step 1
# Get the words from each line in the dataset
words=[line.split('\t')[0] for line in lines]

# step 2
# Define defaultdict of type 'int'
freq=defaultdict(int)


# Count frequency of ocurrence for each word in the dataset
for word in words:
    freq[word]+=1

# step 3
# Create the vocabulary by filtering the 'freq' dictionary
vocab=[k for k,v in freq.items() if (v>1 and k!='\n')]

# step 4
# Sort the vocabulary
vocab.sort()

# Print some random values of the vocabulary
# for i in range(1500,1550):
#     print(vocab[i])

# print(vocab)


with open("./data/prepare.txt", 'w+') as f2:
    context=f2.write(str(vocab))


def assign_unknown(word):
    """
    Assign tokens to unknown words
    """

    # Punctuation characters
    # Try printing them out in a new cell!
    punct = set(string.punctuation)

    # Suffixes
    noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling",
                   "ment", "ness", "or", "ry", "scape", "ship", "ty"]
    verb_suffix = ["ate", "ify", "ise", "ize"]
    adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
    adv_suffix = ["ward", "wards", "wise"]

    # Loop the characters in the word, check if any is a digit
    if any(char.isdigit() for char in word):
        return "--unknown_digit--"

    # Loop the characters in the word, check if any is a punctuation character
    elif any(char in punct for char in word):
        return "--unknown_punct--"

    # Loop the characters in the word, check if any is an upper case character
    elif any(char.isupper() for char in word):
        return "--unknown_upper--"

    # Check if word ends with any noun suffix
    elif any(word.endswith(suffix) for suffix in noun_suffix):
        return "--unknown_noun--"

    # Check if word ends with any verb suffix
    elif any(word.endswith(suffix) for suffix in verb_suffix):
        return "--unknown_verb--"

    # Check if word ends with any adjective suffix
    elif any(word.endswith(suffix) for suffix in adj_suffix):
        return "--unknown_adj--"

    # Check if word ends with any adverb suffix
    elif any(word.endswith(suffix) for suffix in adv_suffix):
        return "--unknown_adv--"

    # If none of the previous criteria is met, return plain unknown
    return "--unknown--"


def get_word_tag(line,vocab):

    if not line.split():
        word="--n--"
        tag="--s--"
    else:
        word,tag=line.split()
        if word not in vocab:
            word=assign_unknown(word)

    return word,tag


print(get_word_tag('\n',vocab))
print(get_word_tag('In\tIN\n',vocab))
print(get_word_tag('tardigrade\tNN\n',vocab))
print(get_word_tag('scrutinize\tVB\n',vocab))




