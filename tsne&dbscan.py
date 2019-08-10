# random.seed =1
from sklearn.manifold import TSNE
import numpy as np
import glob
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

#---------------Data--------------------------------------------------------------
files = glob.glob("6_Output_Cell_Measurement_converted/2_new/*_combine.csv")
data = pd.read_csv(files[0])

for i in range(1,len(files)):
    data = pd.concat([data, pd.read_csv(files[i])], axis=0, ignore_index=True)

for i in range(0,9):
    data.iloc[:,i] = (data.iloc[:,i] - data.iloc[:,i].mean()) / data.iloc[:,i].std()

data.drop(labels='Intensity_Mean', inplace=True, axis=1)

weeks = np.unique(data.loc[:,'Week'])
#-----------------------------------------------------------------------------------

tsne = TSNE(n_components=2)
X_embedded = tsne.fit_transform(data.loc[data['Week'] == 24, :].iloc[:, 0:8])


fig = plt.figure()
fig.set_size_inches(25, 12)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], marker='.')
plt.legend(scatterpoints=1, frameon=False, labelspacing=1)
# plt.legend()
plt.grid()
plt.show()



ns = 10
nbrs = NearestNeighbors(n_neighbors=ns).fit(X_embedded)
distances, indices = nbrs.kneighbors(X_embedded)
distanceDec = sorted(distances[:, ns - 1], reverse=True)
plt.plot(indices[:, 0], distanceDec)
# plt.ylim((2,5))
plt.xlabel('No. of points')
plt.ylabel('eps')
plt.show()
plt.close()

A = []
B = []
C = []

# for i in np.linspace(0.1,10,50):
for i in np.linspace(0.2, 10, 80):

    db = DBSCAN(eps=i, min_samples=11).fit(X_embedded)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    sum = 0
    for t in labels:
        if t == -1:
            sum = sum + 1
    C.append(sum)

    A.append(i)
    B.append(int(n_clusters_))

results = pd.DataFrame([A, B, C]).T
results.columns = ['distance', 'Number of clusters', 'Number of outliers']
results.plot(x='distance', y='Number of clusters', figsize=(10, 6), xlim=(0, 6))
plt.show()
plt.close()

db = DBSCAN(eps=3.5, min_samples=11).fit(X_embedded)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)