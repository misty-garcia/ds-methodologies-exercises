import pandas as pd

import util

def get_data(query, db):
    return pd.read_sql(query, util.get_url(db))

def df_value_counts(df):
    for col in df.columns:
        print(f'{col}:')
        if df[col].dtype == 'object':
            col_count = df[col].value_counts()
        else:
            if df[col].nunique() >= 35:
                col_count = df[col].value_counts(bins=10, sort=False)
            else:
                col_count = df[col].value_counts()
        print(col_count)

def df_feature_dist(df):
    features_num = list(df.select_dtypes(np.number).columns)
    for feature in features_num:
        sns.distplot(df[feature].dropna())
        plt.show()

