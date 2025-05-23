from qdrant_client import models, QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue


# --- --- --- New collection

client = QdrantClient(host="localhost", port=6333)
cname="multi"

def create_collection():
    pass
#

create_collection()


# --- --- --- Load dataset
import pandas as pd
df = pd.read_json("../input.json")

def insert_data():
    pass
#

insert_data()



# --- --- --- NN Search

# Points closest to the first point (id 0)
def get_closest()->list[models.ScoredPoint]:
    return []
#

print("Closest point to the first point")
results = get_closest()
print(results[0])


# Get 2 closest points to the first point by multivector
def get_closest_multi_vector()->list[models.ScoredPoint]:
    return []
#

print("Closest point to the first point by multivector")
results = get_closest_multi_vector()
assert results[0].id == 0 # Itself
print(results)

print()


# --- --- --- Payload update

def update_payload():
    pass
#

update_payload()


# --- --- --- Payload filter

def not_positive_is_positive()->list[models.ScoredPoint]:
    return []
#

print("Positive is Positive is False")
results = not_positive_is_positive()
for r in results:
    print(r)
print()



def not_positive_is_positive_score(threshold:float)->list[models.ScoredPoint]:
    return []
#

print("Positive is Positive is False and score > threshold")
results = not_positive_is_positive_score(0.28)
for r in results:
    print(r)
print()