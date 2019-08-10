import pandas as pd
import numpy as np
import glob
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
files = glob.glob("6_Output_Cell_Measurement_converted/*/*_combine.csv")
files.sort()
df_temp = pd.read_csv(files[0])
for i in range(1,len(files)):
    df_temp = pd.concat([df_temp, pd.read_csv(files[i])], axis=0, ignore_index=True)

for i in range(df_temp.shape[1]):
    try:
        if i != 7:
            temp = float(df_temp.iloc[0,i])
            df_temp.iloc[:,i] = (df_temp.iloc[:,i] - df_temp.iloc[:,i].mean()) / df_temp.iloc[:,i].std()
    except:
        continue

df1 = df_temp.iloc[:,0:8]
pca_model = PCA(n_components=2)
pca_model.fit(df1.iloc[:,0:3])
df1_new = pd.DataFrame(pca_model.transform(df1.iloc[:,0:3]))
print(pca_model.explained_variance_ratio_)
cols = list(df1.columns.values)[:-1]
print(cols)
weeks = np.unique(df1['Week'])
indexes = {}
colors = ['red', 'blue', 'green', 'black']
for i in range(0,len(weeks)):
    indexes[weeks[i]] = df1.loc[df1['Week'] == weeks[i],'Week'].index.values

for i in range(0,len(weeks)):
    plt.scatter(df1_new.iloc[indexes[weeks[i]], 0], df1_new.iloc[indexes[weeks[i]], 1], c=colors[i], label='Week {}'.format(weeks[i]), s=0.5)
plt.legend()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA for FP1 and FP2 combined")
plt.savefig("PCA_for_FP1_FP2_combined.png")
plt.close()

for i in range(0,len(weeks)):
    plt.scatter(df1.loc[df1['Week'] == weeks[i], 'Area'], df1.loc[df1['Week'] == weeks[i], 'Circularity'], c=colors[i], s=0.5, label='Week {}'.format(weeks[i]))
plt.xlabel("Area")
plt.ylabel("Circularity")
plt.legend()
plt.title("Area vs Circularity")
plt.savefig("Area_vs_Circularity_for_FP1_FP2_combined_all_weeks.png")
plt.close()

for i in range(0,len(weeks)):
    plt.scatter(df1.loc[df1['Week'] == weeks[i], 'Area'], df1.loc[df1['Week'] == weeks[i], 'Aspect_Ratio'], c=colors[i], label='Week {}'.format(weeks[i]), s=0.5)
plt.xlabel("Area")
plt.ylabel("Aspect_Ratio")
plt.legend()
plt.title("Area vs Aspect_Ratio for FP1 and FP2 combined")
plt.savefig("Area_vs_AR_for_FP1_FP2_combined_all_weeks.png")
plt.close()

for p in cols:
    for q in cols:
        if p != q:
            for i in range(0,len(weeks)):
                if weeks[i] == 0:
                    plt.scatter(df1.loc[df1['Week'] == 0, p], df1.loc[df1['Week'] == 0, q],c='red', s=5, label='Week 0')
                    plt.scatter(df1.loc[df1['Week'] == 8, p], df1.loc[df1['Week'] == 8, q],c='blue', s=0.5, label='Week 8', alpha=0.5)
                    plt.scatter(df1.loc[df1['Week'] == 16, p], df1.loc[df1['Week'] == 16, q],c='blue', s=0.5, label='Week 16',alpha=0.5)
                    plt.scatter(df1.loc[df1['Week'] == 24, p], df1.loc[df1['Week'] == 24, q],c='blue', s=0.5, label='Week 24',alpha=0.5)
                    plt.legend()
                    plt.xlabel(p)
                    plt.ylabel(q)
                    plt.title("{} vs {} - Week {} vs Rest of the weeks for FP1 and FP2 combined".format(p,q,weeks[i]))
                    plt.savefig("{}_vs_{}_for_FP1_FP2_combined_week_0.png".format(p,q))
                    plt.close()

                if weeks[i] == 8:
                    plt.scatter(df1.loc[df1['Week'] == 8, p], df1.loc[df1['Week'] == 8, q],c='red', s=5, label='Week 8')
                    plt.scatter(df1.loc[df1['Week'] == 0, p], df1.loc[df1['Week'] == 0, q],c='blue', s=0.5, label='Week 0', alpha=0.5)
                    plt.scatter(df1.loc[df1['Week'] == 16, p], df1.loc[df1['Week'] == 16, q],c='blue', s=0.5, label='Week 16',alpha=0.5)
                    plt.scatter(df1.loc[df1['Week'] == 24, p], df1.loc[df1['Week'] == 24, q],c='blue', s=0.5, label='Week 24',alpha=0.5)
                    plt.legend()
                    plt.xlabel(p)
                    plt.ylabel(q)
                    plt.title("{} vs {} - Week {} vs Rest of the weeks for FP1 and FP2 combined".format(p,q,weeks[i]))
                    plt.savefig("{}_vs_{}_for_FP1_FP2_combined_week_8.png".format(p,q))
                    plt.close()

                if weeks[i] == 16:
                    plt.scatter(df1.loc[df1['Week'] == 16, p], df1.loc[df1['Week'] == 16, q],c='red', s=5, label='Week 16')
                    plt.scatter(df1.loc[df1['Week'] == 8, p], df1.loc[df1['Week'] == 8, q],c='blue', s=0.5, label='Week 8', alpha=0.5)
                    plt.scatter(df1.loc[df1['Week'] == 0, p], df1.loc[df1['Week'] == 0, q],c='blue', s=0.5, label='Week 0',alpha=0.5)
                    plt.scatter(df1.loc[df1['Week'] == 24, p], df1.loc[df1['Week'] == 24, q],c='blue', s=0.5, label='Week 24',alpha=0.5)
                    plt.legend()
                    plt.xlabel(p)
                    plt.ylabel(q)
                    plt.title("{} vs {} - Week {} vs Rest of the weeks for FP1 and FP2 combined".format(p,q,weeks[i]))
                    plt.savefig("{}_vs_{}_for_FP1_FP2_combined_week_16.png".format(p,q))
                    plt.close()

                if weeks[i] == 24:
                    plt.scatter(df1.loc[df1['Week'] == 24, p], df1.loc[df1['Week'] == 24, q],c='red', s=5, label='Week 24')
                    plt.scatter(df1.loc[df1['Week'] == 8, p], df1.loc[df1['Week'] == 8, q],c='blue', s=0.5, label='Week 8', alpha=0.5)
                    plt.scatter(df1.loc[df1['Week'] == 16, p], df1.loc[df1['Week'] == 16, q],c='blue', s=0.5, label='Week 16',alpha=0.5)
                    plt.scatter(df1.loc[df1['Week'] == 0, p], df1.loc[df1['Week'] == 0, q],c='blue', s=0.5, label='Week 0',alpha=0.5)
                    plt.legend()
                    plt.xlabel(p)
                    plt.ylabel(q)
                    plt.title("{} vs {} - Week {} vs Rest of the weeks for FP1 and FP2 combined".format(p,q,weeks[i]))
                    plt.savefig("{}_vs_{}_for_FP1_FP2_combined_week_24.png".format(p,q))
                    plt.close()






