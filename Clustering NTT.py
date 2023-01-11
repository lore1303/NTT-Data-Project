import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_excel('merged_Fram3.1.xlsx', sheet_name= 'Clustering NTT')
dataset['product_category_name'] = dataset['product_category_name'].fillna("other")
dataset['product_photo_quantity'] = dataset['product_photo_quantity'].fillna(0)
    
X = dataset.iloc[:, [2, 3, 5]].values
y = dataset.iloc[:, 2].values
y=y.reshape(-1,1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X1 = sc_X.fit_transform(X)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

numClusters= 4
kmeans = KMeans(n_clusters=numClusters).fit(X1)
centroids = kmeans.cluster_centers_

# Predicting the clusters
labels = kmeans.predict(X1)
# Getting the cluster centers
C = kmeans.cluster_centers_

dataset['Kmeans Cluster'] = labels.tolist()
dataset.to_excel('Clustering Dataset5.xlsx')

#transform n variiables to 2 principal components to plot
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_fit = pca.fit(X)
principalComponents = pca_fit.transform(X)
principalDf = pd.DataFrame(data = principalComponents
         , columns = ['principal component 1', 'principal component 2'])

colors =['red','green','blue','cyan']
centroidColor= []
for item in range(numClusters):
  centroidColor.append(colors[item])

dataPointColor=[]
for row in labels:
  dataPointColor.append(colors[row])

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'], 
c=dataPointColor, s=50, alpha=0.5)

C_transformed = pca_fit.transform(C)
plt.scatter(C_transformed[:, 0], C_transformed[:, 1], c=centroidColor, s=200, marker=('x'))
plt.show()
