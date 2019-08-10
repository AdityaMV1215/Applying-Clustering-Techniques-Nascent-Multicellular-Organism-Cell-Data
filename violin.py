import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

weeks = [0, 8, 16, 24]
sns.set(font_scale=1.5)
my_pal = {1:"green", 2:"blue", 3:"orange"}
for week in weeks:
    fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(8, 25), sharex=True)
    df = pd.read_csv("final_analysis_data/clusters_week_{}.csv".format(week))
    df['Cluster'] = df['Cluster'] + 1
    features = df.columns.values[1:9]
    axis = 0
    for feature in features:
        g = sns.violinplot(df['Cluster'], df[feature],inner='box', ax=axes[axis], palette=my_pal)
        if feature == 'Area':
            g.set_ylabel("Area (${\u03bcm^2}$)", rotation='vertical', **{'size': 20})
        else:
            g.set_ylabel(feature, rotation='vertical', **{'size': 20})
        if axis == 0:
            g.set_title('Week {}'.format(week), **{'size': 20})
        axis = axis + 1
        g.set_xlabel('')

    g.set_xlabel('Cluster', **{'size': 20})
    plt.savefig("each_week_violin_plots/week-{}-all-features.png".format(week))
    #plt.show()
    plt.close()

