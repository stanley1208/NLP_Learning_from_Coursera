import numpy as np # Library for linear algebra and math utils
import pandas as pd # Dataframe library

import matplotlib.pyplot as plt # Library for plots
from utils import confidence_ellipse # Function to add confidence ellipses to charts

data=pd.read_csv('bayes_features.csv') # Load the data from the csv file
# print(data.head(5))

# Plot the samples using columns 1 and 2 of the matrix
fig,ax=plt.subplots(figsize=(8,8)) #Create a new figure with a custom size

colors=['red','green'] # Define a color palete
sentiments=['negative','positive']

index=data.index

# Color base on sentiment
for sentiment in data.sentiment.unique():
    ix=index[data.sentiment==sentiment]
    ax.scatter(data.iloc[ix].positive, data.iloc[ix].negative, c=colors[int(sentiment)], s=0.1, marker='*',
               label=sentiments[int(sentiment)])

ax.legend(loc='best')

# Custom limits for this chart
# plt.xlim(-250,0)
# plt.ylim(-250,0)
#
# plt.xlabel("Positive")  # x-axis label
# plt.ylabel("Negative")  # y-axis label
# plt.show()

# Plot the samples using columns 1 and 2 of the matrix
fig,ax=plt.subplots(figsize=(8,8)) #Create a new figure with a custom size

colors=['red','green'] # Define a color palete
sentiments=['negative','positive']

index=data.index

# Color base on sentiment
for sentiment in data.sentiment.unique():
    ix=index[data.sentiment==sentiment]
    ax.scatter(data.iloc[ix].positive, data.iloc[ix].negative, c=colors[int(sentiment)], s=0.1, marker='*',
               label=sentiments[int(sentiment)])


# Custom limits for this chart
plt.xlim(-200,0)
plt.ylim(-200,0)

plt.xlabel("Positive")  # x-axis label
plt.ylabel("Negative")  # y-axis label

data_pos=data[data.sentiment==1]    # Filter only the positive samples
data_neg=data[data.sentiment==0]    # Filter only the negative samples

# Print confidence ellipses of 2 std
confidence_ellipse(data_pos.positive, data_pos.negative, ax, n_std=2, edgecolor='black', label=r'$2\sigma$' )
confidence_ellipse(data_neg.positive, data_neg.negative, ax, n_std=2, edgecolor='orange' )

# Print confidence ellipses of 3 std
confidence_ellipse(data_pos.positive, data_pos.negative, ax, n_std=3, edgecolor='black',linestyle=":" ,label=r'$2\sigma$' )
confidence_ellipse(data_neg.positive, data_neg.negative, ax, n_std=3, edgecolor='orange',linestyle=":" )
ax.legend(loc="best")

plt.show()

data2=data.copy()   # Copy the whole data frame

# The following 2 lines only modify the entries in the data frame where sentiment == 1
data2.negative[data.sentiment == 1] =  data2.negative * 1.5 + 50 # Modify the negative attribute
data2.positive[data.sentiment == 1] =  data2.positive / 1.5 - 50 # Modify the positive attribute

# Plot the samples using columns 1 and 2 of the matrix
fig,ax=plt.subplots(figsize=(8,8)) #Create a new figure with a custom size

colors=['red','green'] # Define a color palete
sentiments=['negative','positive']

index=data2.index

# Color base on sentiment
for sentiment in data2.sentiment.unique():
    ix=index[data2.sentiment==sentiment]
    ax.scatter(data2.iloc[ix].positive, data2.iloc[ix].negative, c=colors[int(sentiment)], s=0.1, marker='*',
               label=sentiments[int(sentiment)])


# Custom limits for this chart
plt.xlim(-200,0)
plt.ylim(-200,0)

plt.xlabel("Positive")  # x-axis label
plt.ylabel("Negative")  # y-axis label

data_pos=data2[data2.sentiment==1]    # Filter only the positive samples
data_neg=data2[data2.sentiment==0]    # Filter only the negative samples


# Print confidence ellipses of 2 std
confidence_ellipse(data_pos.positive, data_pos.negative, ax, n_std=2, edgecolor='black', label=r'$2\sigma$' )
confidence_ellipse(data_neg.positive, data_neg.negative, ax, n_std=2, edgecolor='orange' )

# Print confidence ellipses of 3 std
confidence_ellipse(data_pos.positive, data_pos.negative, ax, n_std=3, edgecolor='black',linestyle=":" ,label=r'$2\sigma$' )
confidence_ellipse(data_neg.positive, data_neg.negative, ax, n_std=3, edgecolor='orange',linestyle=":" )
ax.legend(loc="best")

plt.show()