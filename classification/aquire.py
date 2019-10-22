import pandas as pd
import util

# In a new python module, acquire.py:
def get_data(query,db):
    return pd.read_sql(query, util.get_url(db))

# get_titanic_data: returns the titanic data from the codeup data science database as a pandas data frame.
def get_titanic_data():
    query = "SELECT * FROM passengers"
    db = "titanic_db"
    return pd.read_sql(query, util.get_url(db))

# get_iris_data: returns the data from the iris_db on the codeup data science database as a pandas data frame. The returned data frame should include the actual name of the species in addition to the species_ids.

def get_iris_data():
    query = """
    SELECT * FROM measurements
    JOIN species USING (species_id)
    """
    db = "iris_db"
    return pd.read_sql(query, util.get_url(db))

