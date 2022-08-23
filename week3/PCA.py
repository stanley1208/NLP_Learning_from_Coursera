import numpy as np                         # Linear algebra library
import matplotlib.pyplot as plt            # library for visualization
from sklearn.decomposition import PCA      # PCA library
import pandas as pd                        # Data frame library
import math                                # Library for math functions
import random                              # Library for pseudo random numbers

np.random.seed(1)
n=1 # The amount of the correlation
x=np.random.uniform(1,2,1000)   # Generate 1000 samples from a uniform random variable
y=x.copy()*n # Make y = n * x

# PCA works better if the data is centered
x=x-np.mean(x)  # Center x. Remove its mean
y=y-np.mean(y)  # Center y. Remove its mean

data=pd.DataFrame({'x':x,'y':y})    # Create a data frame with x and y
plt.scatter(data.x,data.y)  # Plot the original correlated data in blue

pca=PCA(n_components=2) # Instantiate a PCA. Choose to get 2 output variables

# Create the transformation model for this data. Internally, it gets the rotation
# matrix and the explained variance
pcaTr=pca.fit(data)

rotatedData=pcaTr.transform(data)   # Transform the data base on the rotation matrix of pcaTr

# # Create a data frame with the new variables. We call these new variables PC1 and PC2
dataPCA=pd.DataFrame(data=rotatedData,columns=['PC1','PC2'])

# Plot the transformed data in orange
plt.scatter(dataPCA.PC1,dataPCA.PC2)
# plt.show()

print('Eigenvectors or principal component: First row must be in the direction of [1, n]')
print(pcaTr.components_)
print()
print('Eigenvalues or explained variance')
print(pcaTr.explained_variance_)
