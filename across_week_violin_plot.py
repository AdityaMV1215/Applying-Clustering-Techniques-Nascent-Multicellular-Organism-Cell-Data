import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

weeks = [8, 16, 24]
df = pd.read_csv("final_analysis_data/clusters_week_{}.csv".format(0))
features = df.columns.values[1:10]
df_temp = df.loc[df['Cluster'] == 0, features]
for week in weeks:
    df = pd.read_csv("final_analysis_data/clusters_week_{}.csv".format(week))
    df_temp = pd.concat([df_temp, df.loc[df['Cluster'] == 0, features]], axis=0, ignore_index=True)

sns.set(font_scale=1.5)
fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(8, 25), sharex=True)
axis = 0
for feature in features[:-1]:
    g = sns.violinplot(df_temp['Week'], df_temp[feature],inner='box', ax=axes[axis])
    if feature == 'Area':
        g.set_ylabel("Area (${\u03bcm^2}$)", rotation='vertical', **{'size':20})
    else:
        g.set_ylabel(feature, rotation='vertical', **{'size':20})
    axis = axis + 1
    g.set_xlabel('')

g.set_xlabel('Week', **{'size':20})
plt.savefig("across-all-weeks-violin-plot-all-features.png")
#plt.show()
plt.close()


weeks1 = [0,8,16,24]
f = 0
for i in range(0,len(weeks1)-1):
    j = i+1
    for feature in features[:-1]:
        rvs1 = df_temp.loc[df_temp['Week'] == weeks1[i], feature]
        rvs2 = df_temp.loc[df_temp['Week'] == weeks1[j], feature]
        print("p value for feature {} while comparing week {} and week {} is {}".format(feature, weeks1[i], weeks1[j],
                                                                                        stats.ttest_ind(rvs1, rvs2)[
                                                                                            1]))