import pandas as pd
from pandas import DataFrame
from utils.calculate import calculate_distance

class Restaurants:
    def __init__(self, path="database/restaurant.csv"):
        self.path = path
        self.data = pd.read_csv(self.path)
        print("Restaurants loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path).dropna()

    def run(self,
            restaurant_name: str,
            ) -> DataFrame:
        """Search for restaurant ."""
        results = self.data[self.data["shopname"] == restaurant_name]
        # results = results[results["date"] == date]
        # if price_order == "asc":
        #     results = results.sort_values(by=["Average Cost"], ascending=True)
        # elif price_order == "desc":
        #     results = results.sort_values(by=["Average Cost"], ascending=False)

        # if rating_order == "asc":
        #     results = results.sort_values(by=["Aggregate Rating"], ascending=True)
        # elif rating_order == "desc":
        #     results = results.sort_values(by=["Aggregate Rating"], ascending=False)
        if len(results) == 0:
            return "There is no restaurant in this city."
        return results

    def run_for_distance(self,
            restaurant_name1: str, restaurant_name2: str
            ) -> DataFrame:
        """Search for restaurant ."""
        longitude1 = self.data[self.data["shopname"] == restaurant_name1]["longitude"]
        longitude2 = self.data[self.data["shopname"] == restaurant_name2]["longitude"]
        latitude1 = self.data[self.data["shopname"] == restaurant_name1]["latitude"]
        latitude2 = self.data[self.data["shopname"] == restaurant_name2]["latitude"]


        return calculate_distance((latitude1, longitude1), (latitude2, longitude2))



if __name__ == '__main__':
    restaurants = Restaurants()

    print(restaurants.run("福缘居酒楼(河坊街店)"))