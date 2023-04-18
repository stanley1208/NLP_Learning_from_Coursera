import numpy as np  # library for array and matrix manipulation
import pprint   # utilities for console printing
from utils_nb import plot_vectors   # helper function to plot vectors
import matplotlib.pyplot as plt # visualization library


pp=pprint.PrettyPrinter(indent=4)   # Instantiate a pretty printer

# def basic_hash_table(value_l,n_buckets):
#     def hash_function(value,n_buckets):
#         return int(value)%n_buckets
#
#     hash_table={i:[] for i in range(n_buckets)} # Initialize all the buckets in the hash table as empty lists
#
#     for value in value_l:
#         hash_value=hash_function(value,n_buckets)   # Get the hash key for the given value
#         hash_table[hash_value].append(value)    # Add the element to the corresponding bucket
#
#     return hash_table
#
# value_l=[100,10,14,17,97]   # Set of values to hash
# hash_table_example=basic_hash_table(value_l,n_buckets=18)
# pp.pprint(hash_table_example)
#
# P=np.array([[1,1]]) # Define a single plane.
# fig,ax1=plt.subplots(figsize=(8,8)) # Create a plot
#
# plot_vectors([P],axes=[2,2],ax=ax1) # Plot the plane P as a vector
#
# # Plot  random points.
# for i in range(0,10):
#     v1=np.array(np.random.uniform(-2,2,2))  # Get a pair of random numbers between -2 and 2
#     side_of_plane=np.sign(np.dot(P,v1.T))
#
#     # Color the points depending on the sign of the result of np.dot(P, point.T)
#     if side_of_plane==1:
#         ax1.plot([v1[0]],[v1[1]],'bo')  # Plot blue points
#     else:
#         ax1.plot([v1[0]], [v1[1]], 'ro') # Plot red points
#
# plt.show()


# P=np.array([[1,2]]) # Define a single plane.
# # Get a new plane perpendicular to P. We use a rotation matrix
# PT=np.dot([[0,1],[-1,0]],P.T).T
# fig,ax1=plt.subplots(figsize=(8,8)) # Create a plot
#
# plot_vectors([P],colors=['b'],axes=[2,2],ax=ax1) # Plot the plane P as a vector

# Plot the plane P as a 2 vectors.
# We scale by 2 just to get the arrows outside the current box
# plot_vectors([PT*4,PT*-4],colors=['k','k'],axes=[4,4],ax=ax1)


# Plot  random points.
# for i in range(0,20):
#     v1=np.array(np.random.uniform(-4,4,2))  # Get a pair of random numbers between -2 and 2
#     side_of_plane=np.sign(np.dot(P,v1.T))
#
#     # Color the points depending on the sign of the result of np.dot(P, point.T)
#     if side_of_plane==1:
#         ax1.plot([v1[0]],[v1[1]],'bo')  # Plot blue points
#     else:
#         ax1.plot([v1[0]], [v1[1]], 'ro') # Plot red points
#
# plt.show()

P=np.array([[1,1]]) # Single plane
v1=np.array([[1,2]])    # Sample point 1
v2=np.array([[-1,1]])   # Sample point 2
v3=np.array([[-2,-1]])  # Sample point 3

print(np.dot(P,v1.T))
print(np.dot(P,v2.T))
print(np.dot(P,v3.T))


def side_of_plane(P,v):
    dotprofuct=np.dot(P,v.T)    # Get the dot product P * v'
    sign_of_dot_product=np.sign(dotprofuct) # The sign of the elements of the dotproduct matrix
    sign_of_dot_product_scaler=sign_of_dot_product.item()   # The value of the first item
    return sign_of_dot_product_scaler


print(side_of_plane(P,v1))  # In which side is [1, 2]
print(side_of_plane(P,v2))  # In which side is [-1, 1]
print(side_of_plane(P,v3))  # In which side is [-2, -1]

P1=np.array([[1,1]])    # First plane 2D
P2=np.array([[-1,1]])   # Second plane 2D
P3=np.array([[-1,-1]])  # Third plane 2D
P_l=[P1,P1,P3]

# Vector to search
v=np.array([[2,2]])

def hash_multi_plane(P_l,v):
    hash_value=0
    for i,P in enumerate(P_l):
        sign=side_of_plane(P,v)
        hash_i=1 if sign>=0 else 0
        hash_value+=2**i*hash_i
    return hash_value

print(hash_multi_plane(P_l,v)) # Find the number of the plane that containes this value

np.random.seed(0)
num_dimensions=2 # is 300 in assignment
num_planes=3 # is 10 in assignment
random_planes_matrix=np.random.normal(
    size=(num_planes,num_dimensions)
)
print(random_planes_matrix)

v=np.array([[2,2]])

# Side of the plane function. The result is a matrix
def side_of_plane_matrix(P,v):
    dotprofuct=np.dot(P,v.T)    # Get the dot product P * v'
    sign_of_dot_product=np.sign(dotprofuct) # The sign of the elements of the dotproduct matrix
    return sign_of_dot_product

sides_1=side_of_plane_matrix(random_planes_matrix,v)
print(sides_1)


def hash_multi_plane_matrix(P, v, num_planes):
    sides_matrix = side_of_plane_matrix(P, v)  # Get the side of planes for P and v
    hash_value = 0
    for i in range(num_planes):
        sign = sides_matrix[i].item()  # Get the value inside the matrix cell
        hash_i = 1 if sign >= 0 else 0
        hash_value += 2 ** i * hash_i  # sum 2^i * hash_i

    return hash_value


print(hash_multi_plane_matrix(random_planes_matrix,v,num_planes))

word_embedding={"I":np.array([1,0,1]),
                "love":np.array([-1,0,1]),
                "learning":np.array([1,0,1])
                }

word_in_document=['I','love','learning','not_a_word']
document_embedding=np.array([0,0,0])
for word in word_in_document:
    document_embedding+=word_embedding.get(word,0)
print(document_embedding)