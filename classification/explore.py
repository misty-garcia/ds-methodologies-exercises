import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

def object_subplots(df):

    features = df.columns [(df.dtypes == object) & (df.nunique() < 5)]
    
    _, ax = plt.subplots(nrows=1, ncols=len(features), figsize=(16,5))

    survival_rate = df.survived.mean()

    for i, feature in enumerate(features):
        sns.barplot(feature, 'survived', data=df, ax=ax[i], alpha=.5)
        ax[i].set_ylabel('Survival Rate')
        ax[i].axhline(survival_rate, ls='--', color='grey')
    return

