import pandas as pd
import requests

# Using the code from the lesson as a guide, create a dataframe named items that has all of the data for items.
base_url = "http://python.zach.lol"
response = requests.get(base_url)
response.json()

response = requests.get(base_url + "/documentation")
response.json().keys()
print(response.json()["status"])
print(response.json()["payload"])

response = requests.get(base_url + "/api/v1/items")
response.json().keys()
response.json()["payload"].keys()
response.json()["payload"]["items"]

items = pd.DataFrame(response.json()["payload"]["items"])
items

# Do the same thing, but for stores.
response = requests.get(base_url + "/api/v1/stores")
response.json()
response.json().keys()
response.json()["payload"].keys()
response.json()["payload"]["stores"]

stores = pd.DataFrame(response.json()["payload"]["stores"])
stores

# Extract the data for sales. There are a lot of pages of data here, so your code will need to be a little more complex. Your code should continue fetching data from the next page until all of the data is extracted.
response = requests.get(base_url + "/api/v1/sales")
data = response.json()
data.keys()
data["payload"].keys()
data["payload"]["max_page"]
data["payload"]["next_page"]
data["payload"]["page"]
data["payload"][1]

response = requests.get(base_url + "/api/v1/sales?page=183")
data = response.json()
data.keys()

maxpage = data["payload"]["max_page"] + 1

sales = pd.DataFrame()
for page in range(1,maxpage):
    response = requests.get(base_url + "/api/v1/sales?page={}".format(page))
    sale_page = pd.DataFrame(response.json()["payload"]["sales"])
    sales = pd.concat([sales,sale_page])

sales

# Save the data in your files to local csv files so that it will be faster to access in the future.
sales.to_csv('/Users/mists/codeup-data-science/ds-methodologies-exercises/time_series/sales.csv')

# Combine the data from your three separate dataframes into one large dataframe.
items.head()
stores.head()
sales.head()
