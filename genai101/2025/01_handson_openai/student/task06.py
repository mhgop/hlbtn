# --- --- --- PROVIDED CODE --- --- ---
import dotenv
from typing import cast, Any

env = dotenv.dotenv_values('.env')
api_key = cast(str, env["OPENAI_API_KEY"])
model_name = cast(str, env["OPENAI_MODEL_NAME"])


table = """\
| Name                 | ID         | Category        | Quantity Available | Unit Price AUD | Total Sales AUD | Last Restock Date | Supplier              |
|----------------------|------------|-----------------|--------------------|----------------|-----------------|-------------------|-----------------------|
| Organic Apples       | FRU-001    | Fruits          | 500                | 1.50           | 2250            | 2024-10-01        | Fresh Farms Co.       |
| Carrots              | VEG-002    | Vegetables      | 300                | 0.75           | 1200            | 2024-10-10        | GreenLeaf Growers     |
| Wheat Flour          | GRN-003    | Grains          | 200                | 2.00           | 1600            | 2024-09-15        | AgroBest Corp.        |
| Free-range Eggs      | EGG-004    | Poultry         | 150                | 0.50           | 500             | 2024-10-05        | Sunrise Farms         |
| Milk (Gallon)        | DRY-005    | Dairy           | 80                 | 3.00           | 720             | 2024-09-28        | Valley Dairy Inc.     |
| Potatoes             | VEG-006    | Vegetables      | 400                | 0.60           | 960             | 2024-10-07        | GreenLeaf Growers     |
| Strawberries         | FRU-007    | Fruits          | 250                | 2.50           | 1875            | 2024-10-12        | BerryGood Farms       |
| Chicken Breast       | MEAT-008   | Meat            | 100                | 5.00           | 1500            | 2024-09-25        | CountryMeat Ltd.      |
| Honey (Jar)          | SUG-009    | Sweeteners      | 120                | 4.00           | 480             | 2024-10-03        | BeePure Farms         |
| Almonds              | NUT-010    | Nuts & Seeds    | 180                | 7.00           | 1260            | 2024-09-20        | NuttyHarvest Inc.     |
| Tomatoes             | VEG-011    | Vegetables      | 350                | 1.20           | 1200            | 2024-10-15        | GreenLeaf Growers     |
| Lettuce              | VEG-012    | Vegetables      | 200                | 1.00           | 500             | 2024-10-08        | GreenLeaf Growers     |
| Blueberries          | FRU-013    | Fruits          | 300                | 3.00           | 2100            | 2024-09-18        | BerryGood Farms       |
| Beef Cuts            | MEAT-014   | Meat            | 80                 | 8.00           | 1920            | 2024-09-27        | CountryMeat Ltd.      |
| Goat Cheese          | DRY-015    | Dairy           | 60                 | 5.00           | 800             | 2024-09-22        | Valley Dairy Inc.     |
| Spinach              | VEG-016    | Vegetables      | 400                | 1.10           | 880             | 2024-10-09        | Fresh Leaf Farms      |
| Maple Syrup (Bottle) | SUG-017    | Sweeteners      | 100                | 10.00          | 1000            | 2024-09-30        | Sweet Sap Co.         |
| Pecans               | NUT-018    | Nuts & Seeds    | 150                | 6.50           | 975             | 2024-09-25        | NuttyHarvest Inc.     |
| Corn                 | GRN-019    | Grains          | 500                | 1.25           | 1250            | 2024-10-10        | GreenLeaf Growers     |
| Peppers              | VEG-020    | Vegetables      | 300                | 1.30           | 780             | 2024-10-13        | Fresh Leaf Farms      |
| Grapes               | FRU-021    | Fruits          | 250                | 2.75           | 1375            | 2024-09-29        | Ap Farm               |
| Oats                 | GRN-022    | Grains          | 400                | 1.80           | 1440            | 2024-10-02        | AgroBest Corp.        |
| Broccoli             | VEG-023    | Vegetables      | 220                | 1.50           | 330             | 2024-10-06        | Fresh Leaf Farms      |
| Garlic               | VEG-024    | Vegetables      | 150                | 1.20           | 180             | 2024-10-11        | GreenLeaf Growers     |
| Raspberries          | FRU-025    | Fruits          | 120                | 3.50           | 420             | 2024-10-04        | BerryGood Farms       |
| Peaches              | FRU-026    | Fruits          | 200                | 2.20           | 1320            | 2024-10-14        | BerryGood Farms       |
| Zucchini             | VEG-027    | Vegetables      | 250                | 1.15           | 287.50          | 2024-10-16        | Fresh Leaf Farms      |
| Barley               | GRN-028    | Grains          | 300                | 1.50           | 450             | 2024-10-05        | AgroBest Corp.        |
| Pork Chops           | MEAT-029   | Meat            | 90                 | 6.00           | 540             | 2024-10-01        | CountryMeat Ltd.      |
| Cottage Cheese       | DRY-030    | Dairy           | 75                 | 2.80           | 210             | 2024-10-12        | Valley Dairy Inc.     |
| Celery               | VEG-031    | Vegetables      | 400                | 1.00           | 400             | 2024-10-08        | GreenLeaf Growers     |
| Blackberries         | FRU-032    | Fruits          | 150                | 3.00           | 450             | 2024-10-11        | BerryGood Farms       |
| Venison Steaks       | MEAT-033   | Meat            | 50                 | 9.50           | 475             | 2024-09-28        | Wild Game Suppliers   |
| Butter (Pound)       | DRY-034    | Dairy           | 60                 | 4.00           | 240             | 2024-10-02        | Valley Dairy Inc.     |
| Chickpeas            | GRN-035    | Grains          | 250                | 1.40           | 350             | 2024-10-09        | AgroBest Corp.        |
| Kale                 | VEG-036    | Vegetables      | 320                | 1.25           | 400             | 2024-10-13        | Fresh Leaf Farms      |
| Quinoa               | GRN-037    | Grains          | 180                | 2.00           | 360             | 2024-10-10        | AgroBest Corp.        |
| Rhubarb              | VEG-038    | Vegetables      | 220                | 1.75           | 385             | 2024-10-12        | GreenLeaf Growers     |
| Apricots             | FRU-039    | Fruits          | 200                | 2.60           | 520             | 2024-09-29        | Ap Farm               |
| Duck Eggs            | EGG-040    | Poultry         | 120                | 0.80           | 96              | 2024-10-07        | Sunrise Farms         |
"""

# --- --- --- CODE TO BE PRODUCED BY THE STUDENT --- --- ---



# --- --- --- TESTING CODE --- --- ---
import json
import sys
import pandas as pd

if __name__ == "__main__":

    try:
        st = sys.argv[1]
        print("Searching for:", st)
        json_string, cached_tokens = call_with(st)
    except IndexError:
        print("Usage: program <search_term>")
        sys.exit(1)
    
    print(f"Number of cached tokens: {cached_tokens}")
    print()

    results = json.loads(json_string)["results"]
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        #
        assert list(df.columns) == ["Name", "ID", "Category", "Quantity Available", "Unit Price AUD", "Total Sales AUD", "Last Restock Date", "Supplier"], "The columns name should match the specified format."
    else:
        print("No results found.")

            