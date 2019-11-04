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
    # get zillow data
    df = acquire.get_data(query, db)
    # keep only the most recent transaction date
    df = df.sort_values("transactiondate", ascending=False).drop_duplicates("parcelid")
    # keep only 2017 values
    df = df [df.transactiondate.str.startswith("2017")]

    # remove all the duplicate id columns
    df.drop(columns = ["typeconstructiontypeid","storytypeid", "propertylandusetypeid", "heatingorsystemtypeid", "buildingclasstypeid","architecturalstyletypeid","airconditioningtypeid","id"], inplace=True)

    # keep single family homes and remove unit counts greater than 1
    df = df [df.propertylandusedesc == "Single Family Residential"]
    df = df [(df.unitcnt != 2) & (df.unitcnt != 3)]

    # remove rows or columns that have 99% null values
    prep.handle_missing_values(df, .5, .5)

    # remove the following columns with not enough info or duplicate info
    df.drop(columns = ["finishedsquarefeet12","buildingqualitytypeid", "fullbathcnt", "propertyzoningdesc", "unitcnt", "heatingorsystemdesc","assessmentyear","regionidcounty", "rawcensustractandblock", "calculatedbathnbr", "propertycountylandusecode"], inplace=True)

    # remove remaining rows with blanks 
    df.dropna(inplace=True)

    # set index as parcelid
    df.set_index("parcelid", inplace=True)
    
    return df

