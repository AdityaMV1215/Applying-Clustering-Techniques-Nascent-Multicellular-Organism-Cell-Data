import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import warnings
warnings.simplefilter("ignore")

#---------------------------------------------------------------------------------

#---------------Data--------------------------------------------------------------
files = glob.glob("6_Output_Cell_Measurement_converted/2_new/*_combine.csv")
data = pd.read_csv(files[0])

for i in range(1,len(files)):
    data = pd.concat([data, pd.read_csv(files[i])], axis=0, ignore_index=True)

for i in range(0,9):
    if i == 6:
        thresh = (2000-data.iloc[:,i].mean())/data.iloc[:,i].std()
    data.iloc[:,i] = (data.iloc[:,i] - data.iloc[:,i].mean()) / data.iloc[:,i].std()

data.drop(labels='Intensity_Mean', inplace=True, axis=1)
#print(data.iloc[:,5])

weeks = np.unique(data.loc[:,'Week'])
#-----------------------------------------------------------------------------------

#------------TSNE-------------------------------------------------------------------
def tsne(n_components, perplexity, X, plot=1, week=0, random_state=0):
    tsne_model = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    X_new_tsne = tsne_model.fit_transform(X)
    if plot:
        plt.scatter(X_new_tsne[:,0], X_new_tsne[:,1], s=2)
        plt.xlabel("TSNE 1")
        plt.ylabel("TSNE 2")
        plt.title("Original Data after TSNE for week {}".format(week))
        plt.show()
        plt.close()
    return tsne_model.kl_divergence_, X_new_tsne
#-----------------------------------------------------------------------------------

#--------DBSCAN---------------------------------------------------------------------
def dbscan(eps=1, X=None, plot=1, week=0):
    dbscan_model = DBSCAN(eps=eps)
    dbscan_model.fit(X)
    return dbscan_model.labels_


#-----------------------------------------------------------------------------------

#--------Main-----------------------------------------------------------------------
epsilon = {0:3.5, 8:3.7, 16:3.5, 24:3.2}
c = ['red', 'green', 'blue', 'black', 'yellow', 'orange', 'pink', 'brown', 'violet', 'indigo']
for week in weeks:
    kld = []
    X = []
    for i in range(0,10):
        kl_divergence, X_new_tsne = tsne(n_components=2, perplexity=30, X = data.loc[data['Week'] == week, :].iloc[:, 0:8], plot=0, week=week, random_state=i)
        kld.append(kl_divergence)
        X.append(X_new_tsne)
    #print("KL divergence for week {} is {}".format(week, kld))
    best_X_new_tsne = X[np.argmin(np.array(kld))]
    this_weeks_data = data.loc[data['Week'] == week, :]
    this_weeks_data.reset_index(inplace=True, drop=True)
    index_high = this_weeks_data.loc[this_weeks_data['Intensity_Std'] >= thresh,:].index.values
    index_low = this_weeks_data.loc[this_weeks_data['Intensity_Std'] < thresh, :].index.values
    print(index_high)
    print(index_low)
    plt.scatter(best_X_new_tsne[index_high, 0], best_X_new_tsne[index_high, 1], s=2, c='red', alpha=0.5, label='High STD')
    plt.scatter(best_X_new_tsne[index_low, 0], best_X_new_tsne[index_low, 1], s=2, c='blue', alpha=0.1, label='Normal STD')
    plt.xlabel("TSNE 1")
    plt.ylabel("TSNE 2")
    plt.title("Original Data after TSNE for week {}".format(week))
    plt.legend()
    plt.savefig("data_points_with_high_intensity_std_after_tsne/week-{}.png".format(week))
    plt.close()



#print(tsne(n_components=2, perplexity=30, X = data.loc[data['Week'] == 0, :].iloc[:, 0:8], plot=1, week=0, random_state=1))


#-----------------------------------------------------------------------------------