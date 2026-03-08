from serpapi import GoogleSearch
import pandas as pd

params = {
  "engine": "google_trends_trending_now",
  "geo": "US",
  "api_key": "9bdb9d8d83a93b38fa2eb16730baa6fe5a1fd399a8d8b5e371ff5bae70ce561d"
}

search = GoogleSearch(params)
results = search.get_dict()
trending_searches = results.get("trending_searches", [])

# Create a list to store formatted data
formatted_data = []

for trend in trending_searches:
    # Extract the main query
    query = trend.get("query")
    
    # Extract categories if they exist, otherwise label as 'General'
    categories_list = trend.get("categories", [])
    category_names = [cat.get("name") for cat in categories_list]
    category_label = ", ".join(category_names) if category_names else "General"
    
    formatted_data.append({
        "Trend": query,
        "Category": category_label,
    })

df = pd.DataFrame(formatted_data)
print(df.size)
df.to_csv("data.csv")


