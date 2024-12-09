import pandas as pd
from pandas import DataFrame
from util.calculate import calculate_distance


class Tools:
    def __init__(self, path=""):
        self.path = path
        self.data = pd.read_csv(self.path)


    def load_db(self):
        self.data = pd.read_csv(self.path).dropna()

    def run(self,
            name: str,
            ) -> DataFrame:
        """Search for attraction ."""
        results = self.data[self.data["shopname"] == name]
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
            return f"There is no {name} in this city."
        return results

    def run_for_distance(self,
                         name1: str, name2: str
                         ) -> float:
        """Search for distance of attractions ."""
        if len(self.data[self.data["shopname"] == name1]) == 0 or len(self.data[self.data["shopname"] == name2]) == 0:
            print(f"There is no {name1} or {name2} in this city.")
            return 0
        longitude1 = self.data[self.data["shopname"] == name1]["longitude"].values[0]
        longitude2 = self.data[self.data["shopname"] == name2]["longitude"].values[0]
        latitude1 = self.data[self.data["shopname"] == name1]["latitude"].values[0]
        latitude2 = self.data[self.data["shopname"] == name2]["latitude"].values[0]

        return calculate_distance((latitude1, longitude1), (latitude2, longitude2))


class Attractions(Tools):
    def __init__(self, path="database/attraction.csv"):
        super().__init__(path)
        print("Attractions loaded.")

    def run(self,
            attraction_name: str,
            ) -> DataFrame:
        """Search for attraction ."""
        return super().run(attraction_name)

    def run_for_distance(self,
            attraction_name1: str, attraction_name2: str
            ) -> float:
        """Search for distance of attractions ."""
        return super().run_for_distance(attraction_name1, attraction_name2)

class Restaurants(Tools):
    def __init__(self, path="database/restaurant.csv"):
        super().__init__(path)
        print("Restaurants loaded.")

    def run(self,
            restaurant_name: str,
            ) -> DataFrame:
        """Search for restaurant ."""

        return super().run(restaurant_name)

    def run_for_distance(self,
            restaurant_name1: str, restaurant_name2: str
            ) -> DataFrame:
        """Search for restaurant ."""
        return super().run_for_distance(restaurant_name1, restaurant_name2)





if __name__ == '__main__':
    attractions = Attractions()

    print(attractions.run("千岛湖梅峰岛景区"))
    print(attractions.run_for_distance("千岛湖梅峰岛景区", "钱塘江"))