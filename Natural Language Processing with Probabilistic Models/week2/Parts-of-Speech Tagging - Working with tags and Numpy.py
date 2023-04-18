import numpy as np
import pandas as pd
import math

# Define tags for Adverb, Noun and To (the preposition) , respectively
tags=["RB","NN","TO"]

# Define 'transition_counts' dictionary
# Note: values are the same as the ones in the assignment
transition_counts = {
    ('NN', 'NN'): 16241,
    ('RB', 'RB'): 2263,
    ('TO', 'TO'): 2,
    ('NN', 'TO'): 5256,
    ('RB', 'TO'): 855,
    ('TO', 'NN'): 734,
    ('NN', 'RB'): 2431,
    ('RB', 'NN'): 358,
    ('TO', 'RB'): 200
}

# Store the number of tags in the 'num_tags' variable
num_tags=len(tags)

# Initialize a 3X3 numpy array with zeros
transition_matrix=np.zeros((num_tags,num_tags))

print(transition_matrix)

print(transition_matrix.shape)

# Create sorted version of the tag's list
sorted_tags=sorted(tags)

print(sorted_tags)

for i in range(num_tags):
    for j in range(num_tags):
        # Define tag pair
        tag_tuple=(sorted_tags[i],sorted_tags[j])
        # Get frequency from transition_counts dict and assign to (i, j) position in the matrix
        transition_matrix[i,j]=transition_counts.get(tag_tuple)


print(transition_matrix)

# Define 'print_matrix' function
def print_matrix(matrix):
    print(pd.DataFrame(matrix,index=sorted_tags,columns=sorted_tags))

print_matrix(transition_matrix)

# Scale transition matrix
transition_matrix=transition_matrix/10

print_matrix(transition_matrix)

# Compute sum of row for each row
rows_sum=transition_matrix.sum(axis=1,keepdims=True)

print(rows_sum)

# Normalize transition matrix
transition_matrix=transition_matrix/rows_sum

print_matrix(transition_matrix)

print(transition_matrix.sum(axis=1,keepdims=True))

t_matrix_for=np.copy(transition_matrix)

t_matrix_np=np.copy(transition_matrix)


# Loop values in the diagonal
for i in range(num_tags):
    t_matrix_for[i,i]=t_matrix_for[i,i]+math.log(rows_sum[i])

print_matrix(t_matrix_for)


# Save diagonal in a numpy array
d=np.diag(t_matrix_np)

print(d.shape)

# Reshape diagonal numpy array
d=np.reshape(d,(3,1))

print(d.shape)

# Perform the vectorized operation
d=d+np.vectorize(math.log)(rows_sum)

# Use numpy's 'fill_diagonal' function to update the diagonal
np.fill_diagonal(t_matrix_np,d)

print_matrix(t_matrix_np)

print(t_matrix_np==t_matrix_for)
