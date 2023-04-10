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


# UNQ_C4 GRADED FUNCTION: create_emission_matrix

def create_emission_matrix(alpha, tag_counts, emission_counts, vocab):
    '''
    Input:
        alpha: tuning parameter used in smoothing
        tag_counts: a dictionary mapping each tag to its respective count
        emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
        vocab: a dictionary where keys are words in vocabulary and value is an index.
               within the function it'll be treated as a list
    Output:
        B: a matrix of dimension (num_tags, len(vocab))
    '''

    # get the number of POS tag
    num_tags = len(tag_counts)

    # Get a list of all POS tags
    all_tags = sorted(tag_counts.keys())

    # Get the total number of unique words in the vocabulary
    num_words = len(vocab)

    # Initialize the emission matrix B with places for
    # tags in the rows and words in the columns
    B = np.zeros((num_tags, num_words))

    # Get a set of all (POS, word) tuples
    # from the keys of the emission_counts dictionary
    emis_keys = set(list(emission_counts.keys()))

    ### START CODE HERE (Replace instances of 'None' with your code) ###

    # Go through each row (POS tags)
    for i in range(num_tags):  # Replace None in this line with the proper range.

        # Go through each column (words)
        for j in range(num_words):  # Replace None in this line with the proper range.

            # Initialize the emission count for the (POS tag, word) to zero
            count = 0

            # Define the (POS tag, word) tuple for this row and column
            key = (all_tags[i],vocab[j])  # tuple of form (tag,word)

            # check if the (POS tag, word) tuple exists as a key in emission counts
            if key in emission_counts:  # Replace None in this line with the proper condition.

                # Get the count of (POS tag, word) from the emission_counts d
                count = emission_counts[key]

            # Get the count of the POS tag
            count_tag = tag_counts[all_tags[i]]

            # Apply smoothing and store the smoothed value
            # into the emission matrix B for this row and column
            B[i, j] = (count+alpha)/(count_tag+alpha*num_words)

    ### END CODE HERE ###
    return B



# creating your emission probability matrix. this takes a few minutes to run.
alpha = 0.001
B = create_emission_matrix(alpha, tag_counts, emission_counts, list(vocab))

print(f"View Matrix position at row 0, column 0: {B[0,0]:.9f}")
print(f"View Matrix position at row 3, column 1: {B[3,1]:.9f}")

# Try viewing emissions for a few words in a sample dataframe
cidx  = ['725','adroitly','engineers', 'promoted', 'synergy']

# Get the integer ID for each word
cols = [vocab[a] for a in cidx]

# Choose POS tags to show in a sample dataframe
rvals =['CD','NN','NNS', 'VB','RB','RP']

# For each POS tag, get the row number from the 'states' list
rows = [states.index(a) for a in rvals]

# Get the emissions for the sample of words, and the sample of POS tags
B_sub = pd.DataFrame(B[np.ix_(rows,cols)], index=rvals, columns = cidx )
print(B_sub)


# UNQ_C5 GRADED FUNCTION: initialize
def initialize(states, tag_counts, A, B, corpus, vocab):
    '''
    Input:
        states: a list of all possible parts-of-speech
        tag_counts: a dictionary mapping each tag to its respective count
        A: Transition Matrix of dimension (num_tags, num_tags)
        B: Emission Matrix of dimension (num_tags, len(vocab))
        corpus: a sequence of words whose POS is to be identified in a list
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output:
        best_probs: matrix of dimension (num_tags, len(corpus)) of floats
        best_paths: matrix of dimension (num_tags, len(corpus)) of integers
    '''
    # Get the total number of unique POS tags
    num_tags = len(tag_counts)

    # Initialize best_probs matrix
    # POS tags in the rows, number of words in the corpus as the columns
    best_probs = np.zeros((num_tags, len(corpus)))

    # Initialize best_paths matrix
    # POS tags in the rows, number of words in the corpus as columns
    best_paths = np.zeros((num_tags, len(corpus)), dtype=int)

    # Define the start token
    s_idx = states.index("--s--")
    ### START CODE HERE (Replace instances of 'None' with your code) ###

    # Go through each of the POS tags
    for i in range(num_tags):  # Replace None in this line with the proper range.

        # Handle the special case when the transition from start token to POS tag i is zero
        if A[s_idx,i]==0:  # Replace None in this line with the proper condition. # POS by word

            # Initialize best_probs at POS tag 'i', column 0, to negative infinity
            best_probs[i, 0] = float("-inf")

        # For all other cases when transition from start token to POS tag i is non-zero:
        else:

            # Initialize best_probs at POS tag 'i', column 0
            # Check the formula in the instructions above
            best_probs[i, 0] = np.log(A[s_idx,i])+np.log(B[i,vocab[corpus[0]]])

    ### END CODE HERE ###
    return best_probs, best_paths


best_probs, best_paths = initialize(states, tag_counts, A, B, prep, vocab)


# UNQ_C6 GRADED FUNCTION: viterbi_forward
def viterbi_forward(A, B, test_corpus, best_probs, best_paths, vocab, verbose=True):
    '''
    Input:
        A, B: The transition and emission matrices respectively
        test_corpus: a list containing a preprocessed corpus
        best_probs: an initilized matrix of dimension (num_tags, len(corpus))
        best_paths: an initilized matrix of dimension (num_tags, len(corpus))
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output:
        best_probs: a completed matrix of dimension (num_tags, len(corpus))
        best_paths: a completed matrix of dimension (num_tags, len(corpus))
    '''
    # Get the number of unique POS tags (which is the num of rows in best_probs)
    num_tags = best_probs.shape[0]

    # Go through every word in the corpus starting from word 1
    # Recall that word 0 was initialized in `initialize()`
    for i in range(1, len(test_corpus)):

        # Print number of words processed, every 5000 words
        if i % 5000 == 0 and verbose:
            print("Words processed: {:>8}".format(i))

        ### START CODE HERE  (Replace instances of 'None' with your code EXCEPT the first 'best_path_i = None')  ###
        # For each unique POS tag that the current word can be
        for j in range(num_tags):  # Replace None in this line with the proper range. # for every pos tag

            # Initialize best_prob for word i to negative infinity
            best_prob_i = float("-inf")

            # Initialize best_path for current word i to None
            best_path_i = None  # Do not replace this None # @KEEPTHIS

            # For each POS tag that the previous word can be:
            for k in range(num_tags):  # Replace None in this line with the proper range.

                # Calculate the probability = None
                # best probs of POS tag k, previous word i-1 +
                # log(prob of transition from POS k to POS j) +
                # log(prob that emission of POS j is word i)
                prob = best_probs[k,i-1]+np.log(A[k,j])+np.log(B[j,vocab[test_corpus[i]]])

                # check if this path's probability is greater than
                # the best probability up to and before this point
                if prob>best_prob_i:  # Replace None in this line with the proper condition.

                    # Keep track of the best probability
                    best_prob_i = prob

                    # keep track of the POS tag of the previous word
                    # that is part of the best path.
                    # Save the index (integer) associated with
                    # that previous word's POS tag
                    best_path_i = k

            # Save the best probability for the
            # given current word's POS tag
            # and the position of the current word inside the corpus
            best_probs[j, i] = best_prob_i

            # Save the unique integer ID of the previous POS tag
            # into best_paths matrix, for the POS tag of the current word
            # and the position of the current word inside the corpus.
            best_paths[j, i] = best_path_i

        ### END CODE HERE ###
    return best_probs, best_paths


# this will take a few minutes to run => processes ~ 30,000 words
best_probs, best_paths = viterbi_forward(A, B, prep, best_probs, best_paths, vocab)


# UNQ_C7 GRADED FUNCTION: viterbi_backward
def viterbi_backward(best_probs, best_paths, corpus, states):
    '''
    This function returns the best path.

    '''
    # Get the number of words in the corpus
    # which is also the number of columns in best_probs, best_paths
    m = best_paths.shape[1]

    # Initialize array z, same length as the corpus
    z = [None] * m

    # Get the number of unique POS tags
    num_tags = best_probs.shape[0]

    # Initialize the best probability for the last word
    best_prob_for_last_word = float('-inf')

    # Initialize pred array, same length as corpus
    pred = [None] * m

    ### START CODE HERE (Replace instances of 'None' with your code) ###
    ## Step 1 ##

    # Go through each POS tag for the last word (last column of best_probs)
    # in order to find the row (POS tag integer ID)
    # with highest probability for the last word
    for k in range(num_tags):  # Replace None in this line with the proper range.

        # If the probability of POS tag at row k
        # is better than the previously best probability for the last word:
        if best_probs[k,-1]>best_prob_for_last_word:  # Replace None in this line with the proper condition.

            # Store the new best probability for the last word
            best_prob_for_last_word = best_probs[k,-1]

            # Store the unique integer ID of the POS tag
            # which is also the row number in best_probs
            z[m - 1] = k

    # Convert the last word's predicted POS tag
    # from its unique integer ID into the string representation
    # using the 'states' list
    # store this in the 'pred' array for the last word
    pred[m - 1] = states[k]

    ## Step 2 ##
    # Find the best POS tags by walking backward through the best_paths
    # From the last word in the corpus to the 0th word in the corpus
    for i in range(m-1,-1, -1):  # Replace None in this line with the proper range.
        # Retrieve the unique integer ID of
        # the POS tag for the word at position 'i' in the corpus
        pos_tag_for_word_i = best_paths[np.argmax(best_probs[:,i]),i]
        print(pos_tag_for_word_i)
        # In best_paths, go to the row representing the POS tag of word i
        # and the column representing the word's position in the corpus
        # to retrieve the predicted POS for the word at position i-1 in the corpus
        z[i - 1] = best_paths[pos_tag_for_word_i,i]

        # Get the previous word's POS tag in string form
        # Use the 'states' list,
        # where the key is the unique integer ID of the POS tag,
        # and the value is the string representation of that POS tag
        pred[i - 1] = states[pos_tag_for_word_i]

    ### END CODE HERE ###
    return pred


# Run and test your function
pred = viterbi_backward(best_probs, best_paths, prep, states)
m=len(pred)
print('The prediction for pred[-7:m-1] is: \n', prep[-7:m-1], "\n", pred[-7:m-1], "\n")
print('The prediction for pred[0:8] is: \n', pred[0:7], "\n", prep[0:7])


print('The third word is:', prep[3])
print('Your prediction is:', pred[3])
print('Your corresponding label y is: ', y[3])


# UNQ_C8 GRADED FUNCTION: compute_accuracy
def compute_accuracy(pred, y):
    '''
    Input:
        pred: a list of the predicted parts-of-speech
        y: a list of lines where each word is separated by a '\t' (i.e. word \t tag)
    Output:

    '''
    num_correct = 0
    total = 0

    # Zip together the prediction and the labels
    for prediction, y in zip(pred, y):
        ### START CODE HERE (Replace instances of 'None' with your code) ###
        # Split the label into the word and the POS tag
        word_tag_tuple = y.split()

        # Check that there is actually a word and a tag
        # no more and no less than 2 items
        if len(word_tag_tuple)!=2:  # Replace None in this line with the proper condition.
            continue

        # store the word and tag separately
        word, tag = word_tag_tuple

        # Check if the POS tag label matches the prediction
        if prediction==tag:  # Replace None in this line with the proper condition.

            # count the number of times that the prediction
            # and label match
            num_correct += 1

        # keep track of the total number of examples (that have valid labels)
        total += 1

        ### END CODE HERE ###
    return num_correct / total


print(f"Accuracy of the Viterbi algorithm is {compute_accuracy(pred, y):.4f}")