import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import acquire
import summarize
import prep

query = """
    SELECT *
        FROM properties_2017
        JOIN predictions_2017 USING (parcelid)
        LEFT JOIN airconditioningtype USING (airconditioningtypeid)
        LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
        LEFT JOIN buildingclasstype USING (buildingclasstypeid)
        LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
        LEFT JOIN propertylandusetype USING (propertylandusetypeid)
        LEFT JOIN storytype USING (storytypeid)
        LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
        WHERE latitude is not null and longitude is not null
        """
db = "zillow"

def wrangle():
    df = acquire.get_data(query, db)
    df = df.sort_values("transactiondate", ascending=False).drop_duplicates("parcelid")
    df = df [df.transactiondate.str.startswith("2017")]
    df.drop(columns = ["typeconstructiontypeid","storytypeid", "propertylandusetypeid", "heatingorsystemtypeid", "buildingclasstypeid","architecturalstyletypeid","airconditioningtypeid","id"], inplace=True)
    df.drop(columns = ["finishedsquarefeet12","pooltypeid7"], inplace=True)
    df = df [df.propertylandusedesc == "Single Family Residential"]
    df = df [(df.unitcnt != 2) | (df.unitcnt != 3)]
    prep.handle_missing_values(df, .1, .1)
    return df

