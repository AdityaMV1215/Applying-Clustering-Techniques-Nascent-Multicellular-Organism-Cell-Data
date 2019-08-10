import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
files = glob.glob("6_Output_Cell_Measurement_converted/2_new/*_combine.csv")
files.sort()
df_temp = pd.read_csv(files[0])
for i in range(1,len(files)):
    df_temp = pd.concat([df_temp, pd.read_csv(files[i])], axis=0, ignore_index=True)

'''for i in range(df_temp.shape[1]):
    try:
        if i != 9:
            temp = float(df_temp.iloc[0,i])
            df_temp.iloc[:,i] = (df_temp.iloc[:,i] - df_temp.iloc[:,i].mean()) / df_temp.iloc[:,i].std()
    except:
        continue'''

df1 = df_temp.iloc[:,0:10]
weeks = np.unique(df1['Week'])
cols = list(df1.columns.values)[:-1]
indexes = {}
colors = ['red', 'blue', 'green', 'black']
for i in range(0,len(weeks)):
    indexes[weeks[i]] = df1.loc[df1['Week'] == weeks[i],'Week'].index.values

for p in cols:
    for q in cols:
        if p != q:
            for i in range(0,len(weeks)):
                if weeks[i] == 0:
                    this_weeks_data = df1.loc[df1['Week'] == 0, :]
                    plt.scatter(this_weeks_data.loc[this_weeks_data['Intensity_Std'] >= 2000, p], this_weeks_data.loc[this_weeks_data['Intensity_Std'] >= 2000, q],c='red', s=8, alpha=0.5,label='Intensity STD above 2000')
                    plt.scatter(this_weeks_data.loc[this_weeks_data['Intensity_Std'] < 2000, p], this_weeks_data.loc[this_weeks_data['Intensity_Std'] < 2000, q],c='blue', s=8, alpha=0.1,label='Intensity STD below 2000')
                    plt.legend()
                    plt.xlabel(p.replace('_', ' '))
                    plt.ylabel(q.replace('_', ' '))
                    plt.title("Week {}".format(weeks[i]))
                    plt.savefig("Plots_NEW_1/{}_vs_{}_high_vs_low_week_0.png".format(p,q))
                    plt.close()

                if weeks[i] == 8:
                    this_weeks_data = df1.loc[df1['Week'] == 8, :]
                    plt.scatter(this_weeks_data.loc[this_weeks_data['Intensity_Std'] >= 2000, p],
                                this_weeks_data.loc[this_weeks_data['Intensity_Std'] >= 2000, q], c='red', s=8,
                                label='Intensity STD above 2000', alpha=0.5)
                    plt.scatter(this_weeks_data.loc[this_weeks_data['Intensity_Std'] < 2000, p],
                                this_weeks_data.loc[this_weeks_data['Intensity_Std'] < 2000, q], c='blue', s=8,
                                label='Intensity STD below 2000', alpha=0.1)
                    plt.legend()
                    plt.xlabel(p.replace('_', ' '))
                    plt.ylabel(q.replace('_', ' '))
                    plt.title("Week {}".format(weeks[i]))
                    plt.savefig("Plots_NEW_1/{}_vs_{}_high_vs_low_week_8.png".format(p, q))
                    plt.close()

                if weeks[i] == 16:
                    this_weeks_data = df1.loc[df1['Week'] == 16, :]
                    plt.scatter(this_weeks_data.loc[this_weeks_data['Intensity_Std'] >= 2000, p],
                                this_weeks_data.loc[this_weeks_data['Intensity_Std'] >= 2000, q], c='red', s=8,
                                label='Intensity STD above 2000', alpha=0.5)
                    plt.scatter(this_weeks_data.loc[this_weeks_data['Intensity_Std'] < 2000, p],
                                this_weeks_data.loc[this_weeks_data['Intensity_Std'] < 2000, q], c='blue', s=8,
                                label='Intensity STD below 2000', alpha=0.1)
                    plt.legend()
                    plt.xlabel(p.replace('_', ' '))
                    plt.ylabel(q.replace('_', ' '))
                    plt.title("Week {}".format(weeks[i]))
                    plt.savefig("Plots_NEW_1/{}_vs_{}_high_vs_low_week_16.png".format(p, q))
                    plt.close()

                if weeks[i] == 24:
                    this_weeks_data = df1.loc[df1['Week'] == 24, :]
                    plt.scatter(this_weeks_data.loc[this_weeks_data['Intensity_Std'] >= 2000, p],
                                this_weeks_data.loc[this_weeks_data['Intensity_Std'] >= 2000, q], c='red', s=8,
                                label='Intensity STD above 2000', alpha=0.5)
                    plt.scatter(this_weeks_data.loc[this_weeks_data['Intensity_Std'] < 2000, p],
                                this_weeks_data.loc[this_weeks_data['Intensity_Std'] < 2000, q], c='blue', s=8,
                                label='Intensity STD below 2000', alpha=0.1)
                    plt.legend()
                    plt.xlabel(p.replace('_', ' '))
                    plt.ylabel(q.replace('_', ' '))
                    plt.title("Week {}".format(weeks[i]))
                    plt.savefig("Plots_NEW_1/{}_vs_{}_high_vs_low_week_24.png".format(p, q))
                    plt.close()






