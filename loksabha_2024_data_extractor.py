import requests
import pandas as pd

RESOURCE_ID = "881d1489-a8fc-46b1-b00d-d4824e08e797"
BASE_URL = "https://data.opencity.in/api/action/datastore_search"

all_rows = []
offset = 0
limit = 1000

print("Downloading...")

while True:
    response = requests.get(BASE_URL, params={
        "resource_id": RESOURCE_ID,
        "limit": limit,
        "offset": offset
    })
    
    data = response.json()
    records = data["result"]["records"]
    
    if not records:
        break
        
    all_rows.extend(records)
    offset += limit
    print(f"Downloaded {len(all_rows)} / 61729 rows...")

df = pd.DataFrame(all_rows)

# Filter only your 5 states
STATES = ["Kerala", "Tamil Nadu", "West Bengal", "Assam", "Puducherry"]
df_filtered = df[df["State/UT Name"].isin(STATES)]

# Save
df_filtered.to_csv("agents/data/ls_2024_assembly_segments.csv", index=False)
print(f"Done! Saved {len(df_filtered)} rows for your 5 states.")