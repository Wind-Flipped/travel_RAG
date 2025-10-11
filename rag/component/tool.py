import pandas as pd
from pandas import DataFrame
from geopy.distance import geodesic


class Tools:
    def __init__(self, path=""):
        self.path = path
        self.data = pd.read_csv(self.path)

    def load_db(self):
        self.data = pd.read_csv(self.path).dropna()

    def calculate_distance(self, coord1, coord2):
        distance = geodesic(coord1, coord2).kilometers
        return round(distance, 2)

    def run(self,
            name: str,
            ) -> DataFrame:
        results = self.data[self.data["name"] == name]

        if len(results) == 0:
            return f"There is no {name} in this city."
        return results.iloc[0]

    def run_for_distance(self,
                         name1: str, name2: str
                         ) -> float:

        if len(self.data[self.data["name"] == name1]) == 0:
            print(f"There is no {name1} in this city.")
            return 1000
        if len(self.data[self.data["name"] == name2]) == 0:
            print(f"There is no {name2} in this city.")
            return 1000
        longitude1 = self.data[self.data["name"] == name1]["longitude"].values[0]
        longitude2 = self.data[self.data["name"] == name2]["longitude"].values[0]
        latitude1 = self.data[self.data["name"] == name1]["latitude"].values[0]
        latitude2 = self.data[self.data["name"] == name2]["latitude"].values[0]

        return self.calculate_distance((latitude1, longitude1), (latitude2, longitude2))


class Attractions(Tools):
    def __init__(self, path="../databases/hangzhou_pois.csv"):
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


if __name__ == '__main__':
    attractions = Attractions()

    print(attractions.run("湖滨"))
    print(attractions.run_for_distance("湖滨", "西湖"))
