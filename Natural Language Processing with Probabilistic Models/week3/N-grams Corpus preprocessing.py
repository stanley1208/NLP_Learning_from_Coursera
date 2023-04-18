import nltk
import re


nltk.download('punkt')

# change the corpus to lowercase
corpus="Learning% makes 'me' happy. I am happe be-cause I am learning! :)"
corpusLower=corpus.lower()

print(corpusLower)


# remove special characters
corpusRemoved=re.sub(r"[^a-zA-Z0-9.?! ]+","",corpus)
print(corpusRemoved)

# split text by a delimiter to array
input_date="Sat May  9 07:33:35 CEST 2020"

# get the date parts in array
data_parts=input_date.split(" ")
print(f"data parts = {data_parts}")

#get the time parts in array
time_parts=data_parts[4].split(":")
print(f"tim parts = {time_parts}")


# tokenize the sentence into an array of words
tokenized_sentence=nltk.word_tokenize(corpusRemoved)
print(f"{corpusRemoved}->{tokenized_sentence}")


# find length of each word in the tokenized sentence
# Create a list with the word lengths using a list comprehension
word_lengths=[(word,len(word)) for word in tokenized_sentence]
print(f'Lengths of the words: \n{word_lengths}')


def sentence_to_trigram(tokenized_sentence):
    for i in range(len(tokenized_sentence)-2):
        trigram=tokenized_sentence[i:i+3]
        print(trigram)


print(f"List all trigrams of sentence: {tokenized_sentence}")
print(sentence_to_trigram(tokenized_sentence))

# get trigram prefix from a 4-gram
fourgram=['i','am','happy','because']
trigram=fourgram[0:-1]
print(trigram)


# when working with trigrams, you need to prepend 2 <s> and append one </s>
n=3

tokenized_sentence_final=["<s>"]*(n-1)+tokenized_sentence+["<e>"]
print(tokenized_sentence_final)
