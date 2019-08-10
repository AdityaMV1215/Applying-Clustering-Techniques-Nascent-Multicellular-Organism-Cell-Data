#------------------Imports---------------------------------------------------------
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
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

    return em_model
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
            week_0.append(em(n_components=k, X=X_new_pca, plot=0).bic(X_new_pca))
        elif week == 8:
            week_8.append(em(n_components=k, X=X_new_pca, plot=0).bic(X_new_pca))
        elif week == 16:
            week_16.append(em(n_components=k, X=X_new_pca, plot=0).bic(X_new_pca))
        elif week == 24:
            week_24.append(em(n_components=k, X=X_new_pca, plot=0).bic(X_new_pca))

plt.plot(n_clusters, week_0)
plt.title("BIC score for different values of k for week 0 after PCA for EM")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-0-PCA-EM.png")
plt.close()

plt.plot(n_clusters, week_8)
plt.title("BIC score for different values of k for week 8 after PCA for EM")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-8-PCA-EM.png")
plt.close()

plt.plot(n_clusters, week_16)
plt.title("BIC score for different values of k for week 16 after PCA for EM")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-16-PCA-EM.png")
plt.close()

plt.plot(n_clusters, week_24)
plt.title("BIC score for different values of k for week 24 after PCA for EM")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-24-PCA-EM.png")
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
            week_0_silhouette.append(silhouette_score(X_new_pca, em(n_components=k, X=X_new_pca, plot=0).predict(X_new_pca)))
        elif week == 8:
            week_8_silhouette.append(silhouette_score(X_new_pca, em(n_components=k, X=X_new_pca, plot=0).predict(X_new_pca)))
        elif week == 16:
            week_16_silhouette.append(silhouette_score(X_new_pca, em(n_components=k, X=X_new_pca, plot=0).predict(X_new_pca)))
        elif week == 24:
            week_24_silhouette.append(silhouette_score(X_new_pca, em(n_components=k, X=X_new_pca, plot=0).predict(X_new_pca)))

plt.plot(n_clusters, week_0_silhouette)
plt.title("Silhouette score for different values of k for week 0 after PCA for EM")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-0-PCA-EM.png")
plt.close()

plt.plot(n_clusters, week_8_silhouette)
plt.title("Silhouette score for different values of k for week 8 after PCA for EM")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-8-PCA-EM.png")
plt.close()

plt.plot(n_clusters, week_16_silhouette)
plt.title("Silhouette score for different values of k for week 16 after PCA for EM")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-16-PCA-EM.png")
plt.close()

plt.plot(n_clusters, week_24_silhouette)
plt.title("Silhouette score for different values of k for week 24 after PCA for EM")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-24-PCA-EM.png")
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
            week_0.append(em(n_components=k, X=X_new_ica, plot=0).bic(X_new_ica))
        elif week == 8:
            week_8.append(em(n_components=k, X=X_new_ica, plot=0).bic(X_new_ica))
        elif week == 16:
            week_16.append(em(n_components=k, X=X_new_ica, plot=0).bic(X_new_ica))
        elif week == 24:
            week_24.append(em(n_components=k, X=X_new_ica, plot=0).bic(X_new_ica))

plt.plot(n_clusters, week_0)
plt.title("BIC score for different values of k for week 0 after ICA for EM")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-0-ICA-EM.png")
plt.close()

plt.plot(n_clusters, week_8)
plt.title("BIC score for different values of k for week 8 after ICA for EM")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-8-ICA-EM.png")
plt.close()

plt.plot(n_clusters, week_16)
plt.title("BIC score for different values of k for week 16 after ICA for EM")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-16-ICA-EM.png")
plt.close()

plt.plot(n_clusters, week_24)
plt.title("BIC score for different values of k for week 24 after ICA for EM")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-24-ICA-EM.png")
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
            week_0_silhouette.append(silhouette_score(X_new_ica, em(n_components=k, X=X_new_ica, plot=0).predict(X_new_ica)))
        elif week == 8:
            week_8_silhouette.append(silhouette_score(X_new_ica, em(n_components=k, X=X_new_ica, plot=0).predict(X_new_ica)))
        elif week == 16:
            week_16_silhouette.append(silhouette_score(X_new_ica, em(n_components=k, X=X_new_ica, plot=0).predict(X_new_ica)))
        elif week == 24:
            week_24_silhouette.append(silhouette_score(X_new_ica, em(n_components=k, X=X_new_ica, plot=0).predict(X_new_ica)))

plt.plot(n_clusters, week_0_silhouette)
plt.title("Silhouette score for different values of k for week 0 after ICA for EM")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-0-ICA-EM.png")
plt.close()

plt.plot(n_clusters, week_8_silhouette)
plt.title("Silhouette score for different values of k for week 8 after ICA for EM")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-8-ICA-EM.png")
plt.close()

plt.plot(n_clusters, week_16_silhouette)
plt.title("Silhouette score for different values of k for week 16 after ICA for EM")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-16-ICA-EM.png")
plt.close()

plt.plot(n_clusters, week_24_silhouette)
plt.title("Silhouette score for different values of k for week 24 after ICA for EM")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-24-ICA-EM.png")
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
            week_0.append(em(n_components=k, X=X_new_rp, plot=0).bic(X_new_rp))
        elif week == 8:
            week_8.append(em(n_components=k, X=X_new_rp, plot=0).bic(X_new_rp))
        elif week == 16:
            week_16.append(em(n_components=k, X=X_new_rp, plot=0).bic(X_new_rp))
        elif week == 24:
            week_24.append(em(n_components=k, X=X_new_rp, plot=0).bic(X_new_rp))

plt.plot(n_clusters, week_0)
plt.title("BIC score for different values of k for week 0 after RP for EM")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-0-RP-EM.png")
plt.close()

plt.plot(n_clusters, week_8)
plt.title("BIC score for different values of k for week 8 after RP for EM")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-8-RP-EM.png")
plt.close()

plt.plot(n_clusters, week_16)
plt.title("BIC score for different values of k for week 16 after RP for EM")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-16-RP-EM.png")
plt.close()

plt.plot(n_clusters, week_24)
plt.title("BIC score for different values of k for week 24 after RP for EM")
plt.xlabel("k")
plt.ylabel("BIC value")
plt.savefig("BIC_&_Silhouette_NEW/BIC-Week-24-RP-EM.png")
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
            week_0_silhouette.append(silhouette_score(X_new_rp, em(n_components=k, X=X_new_rp, plot=0).predict(X_new_rp)))
        elif week == 8:
            week_8_silhouette.append(silhouette_score(X_new_rp, em(n_components=k, X=X_new_rp, plot=0).predict(X_new_rp)))
        elif week == 16:
            week_16_silhouette.append(silhouette_score(X_new_rp, em(n_components=k, X=X_new_rp, plot=0).predict(X_new_rp)))
        elif week == 24:
            week_24_silhouette.append(silhouette_score(X_new_rp, em(n_components=k, X=X_new_rp, plot=0).predict(X_new_rp)))

plt.plot(n_clusters, week_0_silhouette)
plt.title("Silhouette score for different values of k for week 0 after RP for EM")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-0-RP-EM.png")
plt.close()

plt.plot(n_clusters, week_8_silhouette)
plt.title("Silhouette score for different values of k for week 8 after RP for EM")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-8-RP-EM.png")
plt.close()

plt.plot(n_clusters, week_16_silhouette)
plt.title("Silhouette score for different values of k for week 16 after RP for EM")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-16-RP-EM.png")
plt.close()

plt.plot(n_clusters, week_24_silhouette)
plt.title("Silhouette score for different values of k for week 24 after RP for EM")
plt.xlabel("k")
plt.ylabel("Silhouette value")
plt.savefig("BIC_&_Silhouette_NEW/Silhouette-Week-24-RP-EM.png")
plt.close()

#-----------------------------------------------------------------------------------