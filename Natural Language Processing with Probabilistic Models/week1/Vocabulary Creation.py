import re   # regular expression library; for tokenization of words
from collections import Counter # collections library; counter: dict subclass for counting hashable objects
import matplotlib.pyplot as plt # for data visualization


# the tiny corpus of text !
text = 'red pink pink blue blue purple yellow ORANGE BLUE BLUE PINK purple' # 🌈
print(text)
print("string length : ",len(text))


# convert all letters to lower case
text_lowercase=text.lower()
print(text_lowercase)
print("string length : ",len(text_lowercase))

# some regex to tokenize the string to words and return them in a list
words=re.findall(r'\w+',text_lowercase)
print(words)
print("count : ",len(words))

# create vocab
vocab=set(words)
print(vocab)
print("count : ",len(vocab))

# create vocab including word count
counts_a=dict()
for w in words:
    counts_a[w]=counts_a.get(w,0)+1
print(counts_a)
print("count : ",len(counts_a))

# create vocab including word count using collections.Counter
counts_b=dict()
counts_b=Counter(words)
print(counts_b)
print("count : ",len(counts_b))

# barchart of sorted word counts
d={'blue':counts_b['blue'],'pink':counts_b['pink'],'purple':counts_b['purple'],'red':counts_b['red'],'yellow':counts_b['yellow'],'orange':counts_b['orange']}
plt.bar(range(len(d)),list(d.values()),align='center',color=d.keys())
_ = plt.xticks(range(len(d)),list(d.keys()))
plt.show()

print("counts_b : ",counts_b)
print("count : ",len(counts_b))