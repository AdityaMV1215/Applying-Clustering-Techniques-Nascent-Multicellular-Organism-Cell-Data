#------------------Imports---------------------------------------------------------
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
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

    return pca_model

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

    return ica_model

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

    return rp_model

#------------------------------------------------------------------------------------
components = [i for i in range(1,10)]

for week in weeks:
    pca_variance = []
    for n_components in range(1,10):
        pca_model = pca(n_components=n_components, X=data.loc[data['Week'] == week, :].iloc[:,0:9], plot=0)
        pca_variance.append(sum(pca_model.explained_variance_ratio_))

    for i in range(0,len(pca_variance)):
        if pca_variance[i] >= 0.7:
            temp = components[i]
            break
    print("# components corresponding to 70% variance explained = {} for week {}".format(temp, week))
    pca_variance = list(map(lambda x: 100*x, pca_variance))
    plt.plot(components, pca_variance)
    plt.xlabel("No. of Components")
    plt.ylabel("% Variance explained")
    plt.title("No. of Components vs % Variance Explained for PCA for Week {}".format(week))
    plt.show()
    plt.close()