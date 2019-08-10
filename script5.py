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

#for i in range(0,9):
    #data.iloc[:,i] = (data.iloc[:,i] - data.iloc[:,i].mean()) / data.iloc[:,i].std()

weeks = np.unique(data.loc[:,'Week'])
#-----------------------------------------------------------------------------------

cols = list(data.columns.values)[0:9]
for attr in cols:
    for week in weeks:
        sns.distplot(data.loc[data['Week'] == week, attr], hist=False, kde=True, label="Week {}".format(week))
    plt.ylabel("Density")
    plt.title("{} Histogram for different Weeks".format(attr))
    plt.savefig("histograms/{}-histogram-for-different-weeks.png".format(attr))
    plt.close()

