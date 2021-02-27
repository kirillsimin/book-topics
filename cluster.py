from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift

import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


data = pd.read_pickle('./results/coords_and_embeddings.pkl')
print(data)

#KMeans
#k_means = KMeans(n_clusters=25)
#k_means.fit(data['coords'].tolist())
#data['category'] = k_means.labels_


#Mean Shift
clustering = MeanShift().fit(data['coords'].tolist())
print(clustering.labels_)
data['category'] = clustering.labels_


print(data.groupby('category', as_index=False)['word'].nunique().to_string())

data = data.groupby(['book_num', 'category'], as_index=False)['word'].nunique()


fig = plt.figure(figsize=(15,15))
plt.scatter(data['book_num'], data['category'], s=data['word']*5)
plt.savefig('./results/book-topics-scatter-category.png')
