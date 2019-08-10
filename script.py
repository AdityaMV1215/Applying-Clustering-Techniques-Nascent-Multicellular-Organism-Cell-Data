import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

files = glob.glob("6_Output_Cell_Measurement_converted/2_correct/*_combine.csv")
df_temp = pd.read_csv(files[0])

for i in range(1,len(files)):
    df_temp = pd.concat([df_temp, pd.read_csv(files[i])], axis=0, ignore_index=True)

weeks = np.unique(df_temp.loc[:,'Week'])
'''for week in weeks:
    plt.scatter(df_temp.loc[df_temp['Week'] == week, 'Week'], df_temp.loc[df_temp['Week'] == week, 'Estimated_Cell_Number'], label='Week {}'.format(week))

plt.xlabel('Week')
plt.ylabel("Estimated Cell Number")
plt.title("Week vs Estimated Cell Number")
plt.legend()

plt.show()
plt.close()

for week in weeks:
    df_area_week = df_temp.loc[df_temp['Week'] == week,['Area', 'Estimated_Cell_Number']]
    df_aspect_ratio_week = df_temp.loc[df_temp['Week'] == week, ['Aspect_Ratio', 'Estimated_Cell_Number']]
    df_area_week_small = df_area_week.loc[df_area_week['Estimated_Cell_Number'] <= 400, 'Area']
    df_apsect_ratio_small = df_aspect_ratio_week.loc[df_aspect_ratio_week['Estimated_Cell_Number'] <= 400, 'Aspect_Ratio']
    df_area_week_big = df_area_week.loc[df_area_week['Estimated_Cell_Number'] > 600, 'Area']
    df_aspect_ratio_big = df_aspect_ratio_week.loc[df_aspect_ratio_week['Estimated_Cell_Number'] > 600, 'Aspect_Ratio']
    plt.scatter(df_area_week_small, df_apsect_ratio_small, label='Small', alpha=0.35, c='red', s=2)
    plt.scatter(df_area_week_big, df_aspect_ratio_big, label='Big', alpha=0.2, c='blue', s=2)
    plt.xlabel("Area")
    plt.ylabel("Aspect Ratio")
    plt.title("Week {} small and big colonies as per estimated cell number".format(week))
    plt.legend()
    plt.show()
    plt.close()

for week in weeks:
    colonies = np.unique(df_temp.loc[df_temp['Week'] == week, 'Cluster_ID'])
    for colony in colonies:
        df_area_colony = df_temp.loc[df_temp['Week'] == week,['Area', 'Cluster_ID']].loc[df_temp['Cluster_ID'] == colony, 'Area']
        df_aspect_ratio_colony = df_temp.loc[df_temp['Week'] == week,['Aspect_Ratio', 'Cluster_ID']].loc[df_temp['Cluster_ID'] == colony, 'Aspect_Ratio']
        plt.scatter(df_area_colony, df_aspect_ratio_colony, label='Colony {}'.format(colony), alpha=0.5, s=2)

    plt.xlabel("Area")
    plt.ylabel("Aspect Ratio")
    plt.title("Week {} different colonies".format(week))
    plt.legend()
    plt.show()
    plt.close()

'''

cols = df_temp.columns.values
for val in cols[0:7]:
    for week in weeks:
        plt.hist(df_temp.loc[df_temp['Week'] == week, val])
        plt.title("Histogram for {} for Week {}".format(val, week))
        plt.ylabel(val)
        plt.show()
        plt.close()


