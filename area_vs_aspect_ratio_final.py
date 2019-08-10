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

data.drop(labels='Intensity_Mean', inplace=True, axis=1)

weeks = np.unique(data.loc[:,'Week'])
#-----------------------------------------------------------------------------------

for week in weeks:
    this_weeks_data = data.loc[data['Week'] == week, ['Area', 'Aspect_Ratio']]
    not_this_weeks_data = data.loc[data['Week'] != week, ['Area', 'Aspect_Ratio']]
    plt.scatter(this_weeks_data['Area'], this_weeks_data['Aspect_Ratio'], s=2, alpha=0.5, label='Week {}'.format(week), c='red')
    plt.scatter(not_this_weeks_data['Area'], not_this_weeks_data['Aspect_Ratio'], s=2, alpha=0.1, label='Rest of the weeks', c='blue')
    plt.xlabel("Area")
    plt.ylabel("Aspect Ratio")
    plt.legend()
    plt.savefig("area_vs_aspect_ratio_final_presentation/week-{}-vs-rest-of-the-weeks.png".format(week))
    plt.close()