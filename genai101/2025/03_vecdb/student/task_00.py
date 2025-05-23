from qdrant_client import QdrantClient
from qdrant_client import models

# --- --- --- Client and test connection

client = QdrantClient(host="localhost", port=6333)

def show_collections():
    pass
#

show_collections()


# --- --- --- Create collection

cname="holberton"

def create_collection():
    pass
#

create_collection()


# --- --- --- Import data

import pandas as pd
df = pd.read_json("../input.json")
print(df.head(n=3))

def insert_data():
    pass
#

insert_data()


# --- --- --- NN Search

# Get closest to the first point
def get_closest()->list[models.ScoredPoint]:
    return []
#

print("Closest point to the first point")
results = get_closest()
print(results)


# Get 2 closest points to the vector of the first point (first result should be the first vector itself)
def get_closest_vector()->list[models.ScoredPoint]:
    return []
#

print("Closest points to the first vector")
results = get_closest_vector()
assert results[0].id == 0 # Itself
print(results)