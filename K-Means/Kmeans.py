import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

dataset = pd.read_csv("Mall_Customers.csv") 

x=dataset.iloc[:,3:].values

from sklearn.cluster import KMeans

ws = [] 

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(x)  
    ws.append(kmeans.inertia_) 

    
plt.plot(range(1,11),ws)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()