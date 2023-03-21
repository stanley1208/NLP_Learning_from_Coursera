from utils_pos import get_word_tag,preprocess
import pandas as pd
from collections import defaultdict
import math
import numpy as np

# load in the training corpus
with open("./data/WSJ_02-21.pos","r") as f:
    training_corpus=f.readlines()

print("A new items of the training corpus list")
print(training_corpus[0:5])

# read the vocabulary data, split by each line of text, and save the list
with open("./data/hmm_vocab.txt","r") as f:
    voc_l=f.read().split('\n')

print("A few items of the vocabulary list")
print(voc_l[0:50])
print()
print("A few items at the end of the vocabulary list")
print(voc_l[-50:])

# vocab: dictionary that has the index of the corresponding words
vocab={}

# Get the index of the corresponding words.
for i,word in enumerate(sorted(voc_l)):
    vocab[word]=i

print("Vocabulary dictionary, key is the word, value is an unique integer.")
cnt=0
for k,v in vocab.items():
    print(f'{k}:{v}')
    cnt+=1
    if cnt>20:
        break


# load in the test corpus
with open("./data/WSJ_24.pos","r") as f:
    y=f.readlines()

print("A sample of the test corpus")
print(y[0:10])


#corpus without tags, preprocessed
_,prep=preprocess(vocab,"./data/test.words")

print("The length of the preprocessed test corpus: ",len(prep))
print("This is a sample of the test_corpus:" )
print(prep[0:10])


# UNQ_C1 GRADED FUNCTION: create_dictionaries
def create_dictionaries(training_corpus, vocab, verbose=True):
    """
    Input:
        training_corpus: a corpus where each line has a word followed by its tag.
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output:
        emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
        transition_counts: a dictionary where the keys are (prev_tag, tag) and the values are the counts
        tag_counts: a dictionary where the keys are the tags and the values are the counts
    """

    # initialize the dictionaries using defaultdict
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)

    # Initialize "prev_tag" (previous tag) with the start state, denoted by '--s--'
    prev_tag = '--s--'

    # use 'i' to track the line number in the corpus
    i = 0

    # Each item in the training corpus contains a word and its POS tag
    # Go through each word and its tag in the training corpus
    for word_tag in training_corpus:

        # Increment the word_tag count
        i += 1

        # Every 50,000 words, print the word count
        if i % 50000 == 0 and verbose:
            print(f"word count = {i}")

        ### START CODE HERE ###
        # get the word and tag using the get_word_tag helper function (imported from utils_pos.py)
        # the function is defined as: get_word_tag(line, vocab)
        word, tag = get_word_tag(word_tag,vocab)

        # Increment the transition count for the previous word and tag
        transition_counts[(prev_tag, tag)] += 1

        # Increment the emission count for the tag and word
        emission_counts[(tag, word)] += 1

        # Increment the tag count
        tag_counts[tag] += 1

        # Set the previous tag to this tag (for the next iteration of the loop)
        prev_tag = tag

        ### END CODE HERE ###

    return emission_counts, transition_counts, tag_counts



emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab)

# get all the POS states
states = sorted(tag_counts.keys())
print(f"Number of POS tags (number of 'states'): {len(states)}")
print("View these POS tags (states)")
print(states)


# print(transition_counts)
# print(emission_counts)
# print(tag_counts)


print("transition examples: ")
for ex in list(transition_counts.items())[:3]:
    print(ex)
print()

print("emission examples: ")
for ex in list(emission_counts.items())[200:203]:
    print (ex)
print()

print("ambiguous word example: ")
for tup,cnt in emission_counts.items():
    if tup[1] == 'back': print (tup, cnt)


# UNQ_C2 GRADED FUNCTION: predict_pos
def predict_pos(prep, y, emission_counts, vocab, states):
    '''
    Input:
        prep: a preprocessed version of 'y'. A list with the 'word' component of the tuples.
        y: a corpus composed of a list of tuples where each tuple consists of (word, POS)
        emission_counts: a dictionary where the keys are (tag,word) tuples and the value is the count
        vocab: a dictionary where keys are words in vocabulary and value is an index
        states: a sorted list of all possible tags for this assignment
    Output:
        accuracy: Number of times you classified a word correctly
    '''

    # Initialize the number of correct predictions to zero
    num_correct = 0

    # Get the (tag, word) tuples, stored as a set
    all_words = set(emission_counts.keys())

    # Get the number of (word, POS) tuples in the corpus 'y'
    total = len(y)
    for word, y_tup in zip(prep, y):

        # Split the (word, POS) string into a list of two items
        y_tup_l = y_tup.split()

        # Verify that y_tup contain both word and POS
        if len(y_tup_l) == 2:

            # Set the true POS label for this word
            true_label = y_tup_l[1]

        else:
            # If the y_tup didn't contain word and POS, go to next word
            continue

        count_final = 0
        pos_final = ''

        # If the word is in the vocabulary...
        if word in vocab:
            for pos in states:

                ### START CODE HERE (Replace instances of 'None' with your code) ###

                # define the key as the tuple containing the POS and word
                key = (pos,word)

                # check if the (pos, word) key exists in the emission_counts dictionary
                if key in emission_counts:  # Replace None in this line with the proper condition.

                    # get the emission count of the (pos,word) tuple
                    count = emission_counts[key]

                    # keep track of the POS with the largest count
                    if count>count_final:  # Replace None in this line with the proper condition.

                        # update the final count (largest count)
                        count_final = count

                        # update the final POS
                        pos_final = pos

            # If the final POS (with the largest count) matches the true POS:
            if pos_final==true_label:  # Replace None in this line with the proper condition.
                # Update the number of correct predictions
                num_correct += 1

    ### END CODE HERE ###
    accuracy = num_correct / total

    return accuracy


accuracy_predict_pos = predict_pos(prep, y, emission_counts, vocab, states)
print(f"Accuracy of prediction using predict_pos is {accuracy_predict_pos:.4f}")


# UNQ_C3 GRADED FUNCTION: create_transition_matrix
def create_transition_matrix(alpha, tag_counts, transition_counts):
    '''
    Input:
        alpha: number used for smoothing
        tag_counts: a dictionary mapping each tag to its respective count
        transition_counts: a dictionary where the keys are (prev_tag, tag) and the values are the counts
    Output:
        A: matrix of dimension (num_tags,num_tags)
    '''
    # Get a sorted list of unique POS tags
    all_tags=sorted(tag_counts.keys())

    # Count the number of unique POS tags
    num_tags = len(all_tags)

    # Initialize the transition matrix 'A'
    A = np.zeros((num_tags, num_tags))

    # Get the unique transition tuples (previous POS, current POS)
    trans_keys = set(transition_counts.keys())

    ### START CODE HERE ###

    # Go through each row of the transition matrix A
    for i in range(num_tags):

        # Go through each column of the transition matrix A
        for j in range(num_tags):

            # Initialize the count of the (prev POS, current POS) to zero
            count = 0

            # Define the tuple (prev POS, current POS)
            # Get the tag at position i and tag at position j (from the all_tags list)
            key = (all_tags[i],all_tags[j])  # tuple of form (tag,tag)

            # Check if the (prev POS, current POS) tuple
            # exists in the transition counts dictionary
            if key in transition_counts:  # Replace None in this line with the proper condition.

                # Get count from the transition_counts dictionary
                # for the (prev POS, current POS) tuple
                count = transition_counts[key]

                # Get the count of the previous tag (index position i) from tag_counts
            count_prev_tag = tag_counts[all_tags[i]]

            # Apply smoothing using count of the tuple, alpha,
            # count of previous tag, alpha, and total number of tags
            A[i, j] = (count+alpha)/(count_prev_tag+alpha*num_tags)

    ### END CODE HERE ###
    return A



alpha = 0.001
A = create_transition_matrix(alpha, tag_counts, transition_counts)
# Testing your function
print(f"A at row 0, col 0: {A[0,0]:.9f}")
print(f"A at row 3, col 1: {A[3,1]:.4f}")

print("View a subset of transition matrix A")
A_sub = pd.DataFrame(A[30:35,30:35], index=states[30:35], columns = states[30:35] )
print(A_sub)