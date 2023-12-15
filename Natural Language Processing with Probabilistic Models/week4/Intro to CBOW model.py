import numpy as np


# ReLU
np.random.seed(10)
z_1=10*np.random.rand(5,1)-5
print(z_1)
# Create copy of vector and save it in the 'h' variable
h=z_1.copy()
print(h<0)

h[h<0]=0
print(h)

def relu(z):
    result=z.copy()
    result[result<0]=0
    return result


z = np.array([[-1.25459881], [ 4.50714306], [ 2.31993942], [ 0.98658484], [-3.4398136 ]])
print(relu(z))


# Softmax
z=np.array([9,8,11,10,8.5])
print(z)

e_z=np.exp(z)
print(e_z)

sum_e_z=np.sum(e_z)
print(sum_e_z)

print(e_z[0]/sum_e_z)

def softmax(z):
    e_z=np.exp(z)
    sum_e_z=np.sum(e_z)
    return e_z/sum_e_z

print(softmax([9,8,11,10,8.5]))

print(np.sum(softmax([9,8,11,10,8.5]))==1)


# 1-D arrays vs 2-D column vectors
V=5
x_array=np.zeros(V)
print(x_array)
print(x_array.shape)

x_column_vector=x_array.copy()
x_column_vector.shape=(V,1)
print(x_column_vector)
print(x_column_vector.shape)




