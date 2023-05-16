from collections import Counter



# the target size of the vocabulary
M=3

# pre-calculated word counts
# Counter could be used to build this dictionary from the source corpus
word_counts={'happy':5,'because':3,'i':2,'am':2,'learning':3,',':1}
vocabulary=Counter(word_counts).most_common(M)

# remove the frequencies and leave just the words
vocabulary=[w[0] for w in vocabulary]

print(f'the new vocabulary containing {M} most frequent words: {vocabulary}')


# test if words in the input sentences are in the vocabulary, if OOV, print <UNK>
sentence=['am','i','learning']
output_sentence=[]
print(f'input sentence: {sentence}')

for w in sentence:
    if w in vocabulary:
        output_sentence.append(w)
    else:
        output_sentence.append('UNK')

print(f'output_sentence: {output_sentence}')


# iterate through all word counts and print words with given frequency f
f=3

for word,freq in word_counts.items():
    if freq==f:
        print(word)