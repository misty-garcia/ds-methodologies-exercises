import pandas as pd

import util
import acquire

def get_mall():
    return acquire.get_data("SELECT * FROM customers", "mall_customers")
