# encoding: utf-8

"""
Explore the 14 Abnormal classes' Labels and Visualize the Distribution.

"""

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Generate Histogram of the Concurrent Labels Count
def Concurrent_Count_Hist():
    df = pd.read_csv('/gdrive/My Drive/CS598-Project-Data/Data_Entry_2017_v2020.csv')

    label_df = pd.DataFrame({'Label_List': [string.split('|') for string in df['Finding Labels'].values.tolist()] })
    label_df['Concurrent Label Number'] = [len(_list) for _list in label_df['Label_List'].values.tolist()]
    
    ax = sns.countplot(x='Concurrent Label Number', data=label_df)
    ax.set_title('Histogram of Images\' Concurrent Labels Count')

    for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+70))

    plt.rcParams["figure.figsize"] = (15,6)

# Generate Histogram of the Fourteen Labels Count
def Label_Count_Hist():
    df = pd.read_csv('/gdrive/My Drive/CS598-Project-Data/Data_Entry_2017_v2020.csv')

    label_df = pd.DataFrame({'Label_List': [string.split('|') for string in df['Finding Labels'].values.tolist()] })
    label_df['Concurrent Label Number'] = [len(_list) for _list in label_df['Label_List'].values.tolist()]

    labels = pd.DataFrame({'Label': [label for _list in label_df['Label_List'].values.tolist() for label in _list]}) 

    ax = sns.countplot(x='Label', data=labels)
    ax.set_title('Histogram of Fourteen Labels Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+60))

    plt.show()

    print(plt.rcParams["figure.figsize"])


if __name__ == __main__:
    Label_Count_Hist()
    Concurrent_Count_Hist()