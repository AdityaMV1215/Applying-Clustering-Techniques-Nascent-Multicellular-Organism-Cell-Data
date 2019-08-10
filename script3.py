#------------------Imports---------------------------------------------------------
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
import warnings
warnings.simplefilter("ignore")

#---------------------------------------------------------------------------------

#---------------Data--------------------------------------------------------------
files = glob.glob("6_Output_Cell_Measurement_converted/2_new/*_combine.csv")
data = pd.read_csv(files[0])

for i in range(1,len(files)):
    data = pd.concat([data, pd.read_csv(files[i])], axis=0, ignore_index=True)

for i in range(0,9):
    data.iloc[:,i] = (data.iloc[:,i] - data.iloc[:,i].mean()) / data.iloc[:,i].std()

weeks = np.unique(data.loc[:,'Week'])
#-----------------------------------------------------------------------------------

#---------------PCA------------------------------------------------------------------
def pca(n_components, X, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None, plot=1, week=0):
    pca_model = PCA(n_components=n_components, copy=copy, whiten=whiten, svd_solver=svd_solver, tol=tol, iterated_power=iterated_power, random_state=random_state)
    pca_model.fit(X)
    X_new = pca_model.transform(X)
    if plot:
        plt.scatter(X_new[:,1], X_new[:,0])
        plt.title("Week {} after PCA".format(week))
        plt.legend()
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.savefig("Week {}-after-pca.png".format(week))
        plt.close()

    return X_new

#-------------------------------------------------------------------------------------

#-------------ICA---------------------------------------------------------------------
def ica(n_components, X, algorithm='parallel', whiten=True, fun='logcosh', fun_args=None, max_iter=2000, tol=0.000001, w_init=None, random_state=None, plot=1, week=0):
    ica_model = FastICA(n_components=n_components, algorithm=algorithm, whiten=whiten, fun=fun, fun_args=fun_args, max_iter=max_iter, tol=tol, w_init=w_init, random_state=random_state)
    ica_model.fit(X)
    X_new = ica_model.transform(X)
    if plot:
        plt.scatter(X_new[:, 0], X_new[:, 1])
        plt.title("Week {} after ICA".format(week))
        plt.legend()
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.savefig("Week {}-after-ICA.png".format(week))
        plt.close()

    return X_new

#------------------------------------------------------------------------------------

#-----------Random Projection--------------------------------------------------------
def rp(X, n_components='auto', eps=0.1, random_state=None, plot=1, week=0):
    rp_model = GaussianRandomProjection(n_components=n_components, eps=eps, random_state=random_state)
    rp_model.fit(X)
    X_new = rp_model.transform(X)
    if plot:
        plt.scatter(X_new[:, 0], X_new[:, 1])
        plt.title("Week {} after Randomized Projection".format(week))
        plt.legend()
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.savefig("Week {}-after-Random-Projection.png".format(week))
        plt.close()

    return X_new

#------------------------------------------------------------------------------------

#----------------KMeans-------------------------------------------------------------
def kmeans(k, X, init='k-means++', n_init=10, max_iter=3000, tol=0.00001, precompute_distances=True, random_state=None, algorithm='auto', plot=1, comp1=0, comp2=1, week=0, action='none', dimensions=2):
    kmeans_model = KMeans(n_clusters=k, init=init, n_init=n_init, max_iter=max_iter, tol=tol, precompute_distances=precompute_distances, random_state=random_state, algorithm=algorithm)
    kmeans_model.fit(X)
    c = ['red', 'green', 'blue', 'black', 'yellow', 'orange', 'pink', 'brown', 'violet', 'indigo']
    y_new = kmeans_model.predict(X)
    X = np.array(X)
    if plot:
        if action == 'none':
            for i in range(0,k):
                plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c = c[i], label='Cluster {}'.format(i+1))
            plt.title("KMeans on Week {} with {} clusters without Dimensionality Reduction".format(week,k))
            plt.legend()
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.savefig("KMeans-Week {}-without-DR.png".format(week))
            plt.close()

        elif action == 'pca':
            for i in range(0,k):
                plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c = c[i], label='Cluster {}'.format(i+1))
            plt.title("KMeans on Week {} with {} clusters after PCA with {} Principal Components".format(week, k, dimensions))
            plt.legend()
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.savefig("KMeans-Week {}-with-pca.png".format(week))
            plt.close()

        elif action == 'ica':
            for i in range(0,k):
                plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c = c[i], label='Cluster {}'.format(i+1))
            plt.title("KMeans on Week {} with {} clusters after ICA with {} Independent Components".format(week, k, dimensions))
            plt.legend()
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.savefig("KMeans-Week {}-with-ica.png".format(week))
            plt.close()

        elif action == 'rp':
            for i in range(0,k):
                plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c = c[i], label='Cluster {}'.format(i+1))
            plt.title("KMeans on Week {} with {} clusters after Randomized Projection with {} Components".format(week, k, dimensions))
            plt.legend()
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.savefig("KMeans-Week {}-with-rp.png".format(week))
            plt.close()

    return kmeans_model

#------------------------------------------------------------------------------------------------------------------

#---------BIC------------------------------------------------------------------------
def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]],
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)
#------------------------------------------------------------------------------------

#-------------Main Code-------------------------------------------------------------
n_clusters = [i for i in range(2,10)]
week_0 = []
week_8 = []
week_16 = []
week_24 = []
for week in weeks:
    this_weeks_data = data.loc[data['Week'] == week, :].iloc[:,0:9]
    X_new_pca = pca(n_components=2, X=this_weeks_data, plot=0)
    for k in range(2,10):
        if week == 0:
            week_0.append(compute_bic(kmeans(k=k, X=X_new_pca, plot=0), X_new_pca))
        elif week == 8:
            week_8.append(compute_bic(kmeans(k=k, X=X_new_pca, plot=0), X_new_pca))
        elif week == 16:
            week_16.append(compute_bic(kmeans(k=k, X=X_new_pca, plot=0), X_new_pca))
        elif week == 24:
            week_24.append(compute_bic(kmeans(k=k, X=X_new_pca, plot=0), X_new_pca))

plt.plot(n_clusters, week_0)
plt.title("BIC score for different values of k for week 0 after PCA for KMeans")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-0-PCA-KMeans.png")
plt.close()

plt.plot(n_clusters, week_8)
plt.title("BIC score for different values of k for week 8 after PCA for KMeans")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-8-PCA-KMeans.png")
plt.close()

plt.plot(n_clusters, week_16)
plt.title("BIC score for different values of k for week 16 after PCA for KMeans")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-16-PCA-KMeans.png")
plt.close()

plt.plot(n_clusters, week_24)
plt.title("BIC score for different values of k for week 24 after PCA for KMeans")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-24-PCA-KMeans.png")
plt.close()

week_0_silhouette = []
week_8_silhouette = []
week_16_silhouette = []
week_24_silhouette = []
for week in weeks:
    this_weeks_data = data.loc[data['Week'] == week, :].iloc[:,0:9]
    X_new_pca = pca(n_components=2, X=this_weeks_data, plot=0)
    for k in range(2,10):
        if week == 0:
            week_0_silhouette.append(silhouette_score(X_new_pca, kmeans(k=k, X=X_new_pca, plot=0).labels_))
        elif week == 8:
            week_8_silhouette.append(silhouette_score(X_new_pca, kmeans(k=k, X=X_new_pca, plot=0).labels_))
        elif week == 16:
            week_16_silhouette.append(silhouette_score(X_new_pca, kmeans(k=k, X=X_new_pca, plot=0).labels_))
        elif week == 24:
            week_24_silhouette.append(silhouette_score(X_new_pca, kmeans(k=k, X=X_new_pca, plot=0).labels_))

plt.plot(n_clusters, week_0_silhouette)
plt.title("Silhouette score for different values of k for week 0 after PCA for KMeans")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-0-PCA-KMeans.png")
plt.close()

plt.plot(n_clusters, week_8_silhouette)
plt.title("Silhouette score for different values of k for week 8 after PCA for KMeans")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-8-PCA-KMeans.png")
plt.close()

plt.plot(n_clusters, week_16_silhouette)
plt.title("Silhouette score for different values of k for week 16 after PCA for KMeans")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-16-PCA-KMeans.png")
plt.close()

plt.plot(n_clusters, week_24_silhouette)
plt.title("Silhouette score for different values of k for week 24 after PCA for KMeans")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-24-PCA-KMeans.png")
plt.close()


week_0 = []
week_8 = []
week_16 = []
week_24 = []
for week in weeks:
    this_weeks_data = data.loc[data['Week'] == week, :].iloc[:,0:9]
    X_new_ica = ica(n_components=2, X=this_weeks_data, plot=0)
    for k in range(2,10):
        if week == 0:
            week_0.append(compute_bic(kmeans(k=k, X=X_new_ica, plot=0), X_new_ica))
        elif week == 8:
            week_8.append(compute_bic(kmeans(k=k, X=X_new_ica, plot=0), X_new_ica))
        elif week == 16:
            week_16.append(compute_bic(kmeans(k=k, X=X_new_ica, plot=0), X_new_ica))
        elif week == 24:
            week_24.append(compute_bic(kmeans(k=k, X=X_new_ica, plot=0), X_new_ica))

plt.plot(n_clusters, week_0)
plt.title("BIC score for different values of k for week 0 after ICA for KMeans")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-0-ICA-KMeans.png")
plt.close()

plt.plot(n_clusters, week_8)
plt.title("BIC score for different values of k for week 8 after ICA for KMeans")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-8-ICA-KMeans.png")
plt.close()

plt.plot(n_clusters, week_16)
plt.title("BIC score for different values of k for week 16 after ICA for KMeans")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-16-ICA-KMeans.png")
plt.close()

plt.plot(n_clusters, week_24)
plt.title("BIC score for different values of k for week 24 after ICA for KMeans")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-24-ICA-KMeans.png")
plt.close()

week_0_silhouette = []
week_8_silhouette = []
week_16_silhouette = []
week_24_silhouette = []
for week in weeks:
    this_weeks_data = data.loc[data['Week'] == week, :].iloc[:,0:9]
    X_new_ica = ica(n_components=2, X=this_weeks_data, plot=0)
    for k in range(2,10):
        if week == 0:
            week_0_silhouette.append(silhouette_score(X_new_ica, kmeans(k=k, X=X_new_ica, plot=0).labels_))
        elif week == 8:
            week_8_silhouette.append(silhouette_score(X_new_ica, kmeans(k=k, X=X_new_ica, plot=0).labels_))
        elif week == 16:
            week_16_silhouette.append(silhouette_score(X_new_ica, kmeans(k=k, X=X_new_ica, plot=0).labels_))
        elif week == 24:
            week_24_silhouette.append(silhouette_score(X_new_ica, kmeans(k=k, X=X_new_ica, plot=0).labels_))

plt.plot(n_clusters, week_0_silhouette)
plt.title("Silhouette score for different values of k for week 0 after ICA for KMeans")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-0-ICA-KMeans.png")
plt.close()

plt.plot(n_clusters, week_8_silhouette)
plt.title("Silhouette score for different values of k for week 8 after ICA for KMeans")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-8-ICA-KMeans.png")
plt.close()

plt.plot(n_clusters, week_16_silhouette)
plt.title("Silhouette score for different values of k for week 16 after ICA for KMeans")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-16-ICA-KMeans.png")
plt.close()

plt.plot(n_clusters, week_24_silhouette)
plt.title("Silhouette score for different values of k for week 24 after ICA for KMeans")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-24-ICA-KMeans.png")
plt.close()


week_0 = []
week_8 = []
week_16 = []
week_24 = []
for week in weeks:
    this_weeks_data = data.loc[data['Week'] == week, :].iloc[:,0:9]
    X_new_rp = rp(n_components=2, X=this_weeks_data, plot=0)
    for k in range(2,10):
        if week == 0:
            week_0.append(compute_bic(kmeans(k=k, X=X_new_rp, plot=0), X_new_rp))
        elif week == 8:
            week_8.append(compute_bic(kmeans(k=k, X=X_new_rp, plot=0), X_new_rp))
        elif week == 16:
            week_16.append(compute_bic(kmeans(k=k, X=X_new_rp, plot=0), X_new_rp))
        elif week == 24:
            week_24.append(compute_bic(kmeans(k=k, X=X_new_rp, plot=0), X_new_rp))

plt.plot(n_clusters, week_0)
plt.title("BIC score for different values of k for week 0 after RP for KMeans")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-0-RP-KMeans.png")
plt.close()

plt.plot(n_clusters, week_8)
plt.title("BIC score for different values of k for week 8 after RP for KMeans")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-8-RP-KMeans.png")
plt.close()

plt.plot(n_clusters, week_16)
plt.title("BIC score for different values of k for week 16 after RP for KMeans")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-16-RP-KMeans.png")
plt.close()

plt.plot(n_clusters, week_24)
plt.title("BIC score for different values of k for week 24 after RP for KMeans")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-24-RP-KMeans.png")
plt.close()

week_0_silhouette = []
week_8_silhouette = []
week_16_silhouette = []
week_24_silhouette = []
for week in weeks:
    this_weeks_data = data.loc[data['Week'] == week, :].iloc[:,0:9]
    X_new_rp = rp(n_components=2, X=this_weeks_data, plot=0)
    for k in range(2,10):
        if week == 0:
            week_0_silhouette.append(silhouette_score(X_new_rp, kmeans(k=k, X=X_new_rp, plot=0).labels_))
        elif week == 8:
            week_8_silhouette.append(silhouette_score(X_new_rp, kmeans(k=k, X=X_new_rp, plot=0).labels_))
        elif week == 16:
            week_16_silhouette.append(silhouette_score(X_new_rp, kmeans(k=k, X=X_new_rp, plot=0).labels_))
        elif week == 24:
            week_24_silhouette.append(silhouette_score(X_new_rp, kmeans(k=k, X=X_new_rp, plot=0).labels_))

plt.plot(n_clusters, week_0_silhouette)
plt.title("Silhouette score for different values of k for week 0 after RP for KMeans")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-0-RP-KMeans.png")
plt.close()

plt.plot(n_clusters, week_8_silhouette)
plt.title("Silhouette score for different values of k for week 8 after RP for KMeans")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-8-RP-KMeans.png")
plt.close()

plt.plot(n_clusters, week_16_silhouette)
plt.title("Silhouette score for different values of k for week 16 after RP for KMeans")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-16-RP-KMeans.png")
plt.close()

plt.plot(n_clusters, week_24_silhouette)
plt.title("Silhouette score for different values of k for week 24 after RP for KMeans")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-24-RP-KMeans.png")
plt.close()

#-----------------------------------------------------------------------------------