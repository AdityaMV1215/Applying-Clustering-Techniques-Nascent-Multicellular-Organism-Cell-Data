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
dcn = {0:0, 8:0, 16:0, 24:0}
for week in weeks:
    temp = data.loc[data['Week'] == week, ['Detected_Cell_Number', 'Cluster_ID']]
    colonies = np.unique(temp['Cluster_ID'])
    for colony in colonies:
        dcn[week] = dcn[week] + temp.loc[temp['Cluster_ID'] == colony, 'Detected_Cell_Number'].iloc[0]
for week in weeks:
    this_weeks_data = data.loc[data['Week'] == week, :]
    colonies = np.unique(this_weeks_data['Cluster_ID'])
    print(len(colonies))
    for colony in colonies:
        plt.scatter(this_weeks_data['Week'], this_weeks_data['Detected_Cell_Number'], label='Colony {}'.format(colony), facecolors='none', edgecolors=np.random.rand(3,))

plt.ylabel("Detected Cell Number Per Colony", {'size':11})
plt.xticks(ticks=[0,8,16,24], labels=['Week 0\n14 Colonies\n{} Cells'.format(dcn[0]),'Week 8\n15 Colonies\n{} Cells'.format(dcn[8]),'Week 16\n7 Colonies\n{} Cells'.format(dcn[16]),'Week 24\n23 Colonies\n{} Cells'.format(dcn[24])], **{'size':10})
plt.axis(xmin=-6, xmax=30)
plt.savefig("summary-plot.png")
#plt.show()
plt.close()