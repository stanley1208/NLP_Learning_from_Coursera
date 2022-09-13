import numpy as np
import matplotlib.pyplot as plt
from utils_nb import plot_vectors


# Example 1
# Create a 2 x 2 matrix
R=np.array([[2,0],
            [0,-2]])

x=np.array([[1,1]]) # Create a 1 x 2 matrix

y=np.dot(x,R)

print(y)
plot_vectors([x],axes=[4,4],fname='transform_x.svg')
plot_vectors([x,y],axes=[4,4],fname='transformx_and_y.svg')

angle=100*(np.pi/180) #convert degrees to radians

Ro=np.array([[np.cos(angle),-np.sin(angle)],
             [np.sin(angle),np.cos(angle)]])

x2=np.array([2,2]).reshape(1,-1) # make it a row vector
y2=np.dot(x2,Ro)

print('Rotation matrix')
print(Ro)
print('\nRotated vector')
print(y2)

print('\n x2 norm',np.linalg.norm(x2))
print('\n y2 norm',np.linalg.norm(y2))
print('\n Rotation matrix norm',np.linalg.norm(Ro))

plot_vectors([x2,y2],fname='transform_02.svg')

A=np.array([[2,2],
            [2,2]])
A_squared=np.square(A)
print(A_squared)

A_Forbenius=np.sqrt(np.sum(A_squared))
print(A_Forbenius)
print(np.linalg.norm(A_Forbenius))

print('Forbenius norm of the Rotation matrix')
print(np.sqrt(np.sum(Ro*Ro)),'==',np.linalg.norm(Ro))



