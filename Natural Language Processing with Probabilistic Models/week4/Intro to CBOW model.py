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