import pandas as pd
from pandas import DataFrame
from ..utils.calculate import calculate_distance

class Attractions:
    def __init__(self, path="database/attraction.csv"):
        self.path = path
        self.data = pd.read_csv(self.path)
        print("Attractions loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path).dropna()

    def run(self,
            attraction_name: str,
            ) -> DataFrame:
        """Search for attraction ."""
        results = self.data[self.data["shopname"] == attraction_name]
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
            return "There is no attraction in this city."
        return results

    def run_for_distance(self,
            attraction_name1: str, attraction_name2: str
            ) -> DataFrame:
        """Search for distance of attractions ."""
        longitude1 = self.data[self.data["shopname"] == attraction_name1]["longitude"]
        longitude2 = self.data[self.data["shopname"] == attraction_name2]["longitude"]
        latitude1 = self.data[self.data["shopname"] == attraction_name1]["latitude"]
        latitude2 = self.data[self.data["shopname"] == attraction_name2]["latitude"]

        return calculate_distance((latitude1, longitude1), (latitude2, longitude2))


if __name__ == '__main__':
    attractions = Attractions()

    print(attractions.run("千岛湖梅峰岛景区"))
    print(attractions.run_for_distance("千岛湖梅峰岛景区", "钱塘江"))