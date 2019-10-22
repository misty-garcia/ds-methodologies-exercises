import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

def prep_iris(df):
    df.drop(columns = ["species_id","measurement_id"],inplace=True)
    df.rename(columns={"species_name":"species"}, inplace=True)
    encoder = LabelEncoder()
    
    encoder.fit(df.species)
    df.species = encoder.transform(df.species)
    return df

def prep_titanic(df):
    df.fillna(np.nan,inplace=True)

    imp_mode = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    imp_mode.fit(df[["embarked","embark_town"]])
    df[["embarked","embark_town"]] = imp_mode.transform(df[["embarked","embark_town"]])
    
    df.drop(columns="deck", inplace=True)
    
    encoder = LabelEncoder()
    df.embarked = encoder.fit_transform(df.embarked)
    scaler = MinMaxScaler()
    df[["age","fare"]] = scaler.fit_transform(df[["age","fare"]])
    return df