import pandas as pd

import util
import acquire

def get_mall():
    return acquire.get_data("SELECT * FROM customers", "mall_customers")

def wrangle():
    df = get_mall()
    df.set_index("customer_id", inplace=True)
    return df
