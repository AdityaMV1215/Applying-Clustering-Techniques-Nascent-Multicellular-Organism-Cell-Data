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

data.drop(labels='Intensity_Mean', inplace=True, axis=1)

weeks = np.unique(data.loc[:,'Week'])
#-----------------------------------------------------------------------------------

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

    return silhouette_score(X,y_new)

#------------------------------------------------------------------------------------------------------------------

#-------------EM----------------------------------------------------------------------
def em(n_components, X, covariance_type='full', tol=0.000001, reg_covar=0.000001, max_iter=3000, n_init=100, init_params='kmeans', random_state=None, warm_start=True, plot=1, comp1=0, comp2=1, week=0, action='none', dimensions=0):
    em_model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, tol=tol, reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params, random_state=random_state, warm_start=warm_start)
    em_model.fit(X)
    c = ['red', 'green', 'blue', 'black', 'yellow', 'orange', 'pink', 'brown', 'violet', 'indigo']
    y_new = em_model.predict(X)
    X = np.array(X)
    if plot:
        if action == 'none':
            for i in range(0, n_components):
                plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c=c[i], label='Cluster {}'.format(i+1))
            plt.title("EM on Week {} with {} clusters without Dimensionality Reduction".format(week, n_components))
            plt.legend()
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.savefig("EM-Week {}-without-DR.png".format(week))
            plt.close()

        elif action == 'pca':
            for i in range(0, n_components):
                plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c=c[i], label='Cluster {}'.format(i+1))
            plt.title("EM on Week {} with {} clusters after PCA with {} Principal Components".format(week, n_components, dimensions))
            plt.legend()
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.savefig("EM-Week {}-with-pca.png".format(week))
            plt.close()

        elif action == 'ica':
            for i in range(0, n_components):
                plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c=c[i], label='Cluster {}'.format(i+1))
            plt.title("EM on Week {} with {} clusters after ICA with {} Independent Components".format(week, n_components, dimensions))
            plt.legend()
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.savefig("EM-Week {}-with-ica.png".format(week))
            plt.close()

        elif action == 'rp':
            for i in range(0, n_components):
                plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c=c[i], label='Cluster {}'.format(i+1))
            plt.title("EM on Week {} with {} clusters after Randomized Projection with {} Components".format(week, n_components, dimensions))
            plt.legend()
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.savefig("EM-Week {}-with-rp.png".format(week))
            plt.close()

    return silhouette_score(X, y_new)
#------------------------------------------------------------------------------------

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

#--------Main Code-------------------------------------------------------------------
'''min_clusters = 2
max_clusters = 10
min_dimensions = 2
max_dimensions = 7
kmeans_without_dr = {}
em_without_dr = {}
kmeans_pca = {}
kmeans_ica = {}
kmeans_rp = {}
em_pca = {}
em_ica = {}
em_rp = {}
for week in weeks:
    this_weeks_data = data.loc[data['Week'] == week, :].iloc[:,0:7]
    kmeans_without_dr[week] = {}
    em_without_dr[week] = {}
    kmeans_pca[week] = {}
    kmeans_ica[week] = {}
    kmeans_rp[week] = {}
    em_pca[week] = {}
    em_ica[week] = {}
    em_rp[week] = {}
    for n_dimensions in range(min_dimensions, max_dimensions+1):
        for n_clusters in range(min_clusters, max_clusters+1):
            kmeans_without_dr[week][(n_dimensions, n_clusters)] = kmeans(n_clusters, this_weeks_data, plot=0)

            X_new_pca = pca(n_dimensions, this_weeks_data, plot=0)
            kmeans_pca[week][(n_dimensions, n_clusters)] = kmeans(n_clusters, X_new_pca, plot=0)

            X_new_ica = ica(n_dimensions, this_weeks_data, plot=0)
            kmeans_ica[week][(n_dimensions, n_clusters)] = kmeans(n_clusters, X_new_ica, plot=0)

            X_new_rp = rp(this_weeks_data, n_dimensions, plot=0)
            kmeans_rp[week][(n_dimensions, n_clusters)] = kmeans(n_clusters, X_new_rp, plot=0)

            em_without_dr[week][(n_dimensions, n_clusters)] = em(n_clusters, this_weeks_data, plot=0)

            em_pca[week][(n_dimensions, n_clusters)] = em(n_clusters, X_new_pca, plot=0)

            em_ica[week][(n_dimensions, n_clusters)] = em(n_clusters, X_new_ica, plot=0)

            em_rp[week][(n_dimensions, n_clusters)] = em(n_clusters, X_new_rp, plot=0)


kmeans_wdr_week = {}
kmeans_pca_week = {}
kmeans_ica_week = {}
kmeans_rp_week = {}
em_wdr_week = {}
em_pca_week = {}
em_ica_week = {}
em_rp_week = {}

for week in kmeans_without_dr:
    max_key_kmeans_wdr = None
    max_val_kmeans_wdr = -2
    for key in kmeans_without_dr[week]:
        if kmeans_without_dr[week][key] > max_val_kmeans_wdr:
            max_val_kmeans_wdr = kmeans_without_dr[week][key]
            max_key_kmeans_wdr = key
    kmeans_wdr_week[week] = max_key_kmeans_wdr

for week in kmeans_pca:
    max_key_kmeans_pca = None
    max_val_kmeans_pca = -2
    for key in kmeans_pca[week]:
        if kmeans_pca[week][key] > max_val_kmeans_pca:
            max_val_kmeans_pca = kmeans_pca[week][key]
            max_key_kmeans_pca = key
    kmeans_pca_week[week] = max_key_kmeans_pca

for week in kmeans_ica:
    max_key_kmeans_ica = None
    max_val_kmeans_ica = -2
    for key in kmeans_ica[week]:
        if kmeans_ica[week][key] > max_val_kmeans_ica:
            max_val_kmeans_ica = kmeans_ica[week][key]
            max_key_kmeans_ica = key
    kmeans_ica_week[week] = max_key_kmeans_ica

for week in kmeans_rp:
    max_key_kmeans_rp = None
    max_val_kmeans_rp = -2
    for key in kmeans_rp[week]:
        if kmeans_rp[week][key] > max_val_kmeans_rp:
            max_val_kmeans_rp = kmeans_rp[week][key]
            max_key_kmeans_rp = key
    kmeans_rp_week[week] = max_key_kmeans_rp

for week in em_without_dr:
    max_key_em_wdr = None
    max_val_em_wdr = -2
    for key in em_without_dr[week]:
        if em_without_dr[week][key] > max_val_em_wdr:
            max_val_em_wdr = em_without_dr[week][key]
            max_key_em_wdr = key
    em_wdr_week[week] = max_key_em_wdr

for week in em_pca:
    max_key_em_pca = None
    max_val_em_pca = -2
    for key in em_pca[week]:
        if em_pca[week][key] > max_val_em_pca:
            max_val_em_pca = em_pca[week][key]
            max_key_em_pca = key
    em_pca_week[week] = max_key_em_pca

for week in em_ica:
    max_key_em_ica = None
    max_val_em_ica = -2
    for key in em_ica[week]:
        if em_ica[week][key] > max_val_em_ica:
            max_val_em_ica = em_ica[week][key]
            max_key_em_ica = key
    em_ica_week[week] = max_key_em_ica

for week in em_rp:
    max_key_em_rp = None
    max_val_em_rp = -2
    for key in em_rp[week]:
        if em_rp[week][key] > max_val_em_rp:
            max_val_em_rp = em_rp[week][key]
            max_key_em_rp = key
    em_rp_week[week] = max_key_em_rp
    
'''

for week in weeks:
    this_weeks_data = data.loc[data['Week'] == week, :].iloc[:,0:9]

    nd_wdr_kmeans, nc_wdr_kmeans = 5, 2
    kmeans(nc_wdr_kmeans, this_weeks_data, week=week, action='none', dimensions=7)

    nd_pca_kmeans, nc_pca_kmeans = 5, 2
    X_new_pca_kmeans = pca(nd_pca_kmeans, this_weeks_data, week=week)
    kmeans(nc_pca_kmeans, X_new_pca_kmeans, week=week, action='pca', dimensions=nd_pca_kmeans)

    nd_ica_kmeans, nc_ica_kmeans = 5, 2
    X_new_ica_kmeans = ica(nd_ica_kmeans, this_weeks_data, week=week)
    kmeans(nc_ica_kmeans, X_new_ica_kmeans, week=week, action='ica', dimensions=nd_ica_kmeans)

    nd_rp_kmeans, nc_rp_kmeans = 5, 2
    X_new_rp_kmeans = rp(this_weeks_data, nd_rp_kmeans, week=week)
    kmeans(nc_rp_kmeans, X_new_rp_kmeans, week=week, action='rp', dimensions=nd_rp_kmeans)

    nd_wdr_em, nc_wdr_em = 5, 2
    em(nc_wdr_em, this_weeks_data, week=week, action='none', dimensions=7)

    nd_pca_em, nc_pca_em = 5, 2
    X_new_pca_em = pca(nd_pca_em, this_weeks_data, week=week)
    em(nc_pca_em, X_new_pca_em, week=week, action='pca', dimensions=nd_pca_em)

    nd_ica_em, nc_ica_em = 5, 2
    X_new_ica_em = ica(nd_ica_em, this_weeks_data, week=week)
    em(nc_ica_em, X_new_ica_em, week=week, action='ica', dimensions=nd_ica_em)

    nd_rp_em, nc_rp_em = 5, 2
    X_new_rp_em = rp(this_weeks_data, nd_rp_em, week=week)
    em(nc_rp_em, X_new_rp_em, week=week, action='rp', dimensions=nd_rp_em)

#------------------------------------------------------------------------------------