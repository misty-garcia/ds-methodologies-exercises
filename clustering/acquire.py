import pandas as pd

import util

def get_data(query, db):
    return pd.read_sql(query, util.get_url(db))

