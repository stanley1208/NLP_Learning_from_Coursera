# data
word='dearz'  # 🦌


# splits with a loop
splits_a=[]
for i in range(len(word)+1):
    splits_a.append([word[:i],word[i:]])

for i in splits_a:
    print(i)

# same splits, done using a list comprehension
# splits_b=[(word[:i],word[i:]) for i in range(len(word)+1)]
#
# for i in splits_b:
#     print(i)

# deletes with a loop
splits=splits_a
deletes=[]

print("word :",word)
for L,R in splits:
    if R:
        print(L+R[1:],' <-- delete ',R[0])

# breaking it down
print('word :',word)
one_split=splits[0]
print('first item from the splits list :',one_split)
L=one_split[0]
R=one_split[1]
print("L :",L)
print("R :",R)
print('*** now implicit delete by excluding the leading letter ***')
print('L + R[1:] : ',L + R[1:], ' <-- delete ', R[0])

# deletes with a list comprehension
splits=splits_a
deletes=[L+R[1:] for L,R in splits if R]
print(deletes)
print('*** which is the same as ***')
for i in deletes:
    print(i)


vocab = ['dean','deer','dear','fries','and','coke']
edits=list(deletes)

print("vocab :",vocab)
print("edits :",edits)

candidates=[]

### START CODE HERE ###
candidates = set.intersection(set(vocab),edits)  # hint: 'set.intersection'/
### END CODE HERE ###

print('candidate words : ', candidates)

