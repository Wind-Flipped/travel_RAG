import random

import pandas as pd
from pandas import DataFrame
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "./")))
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
        results = self.data[self.data['shopname'] == name]
        if len(results) == 0:
            return f"There is no {name} in this city."
        return results.iloc[0].dropna().to_dict()

    # Get nearest k POIs
    def get_nearest_restaurants(self, longitude: float, latitude: float, k: int = 3):
        df = self.data
        df['distance'] = df.apply(lambda row: calculate_distance((latitude, longitude), (row['latitude'], row['longitude'])), axis=1)
        nearest_pois = df.nsmallest(k, 'distance')

        return nearest_pois

    def get_coordinate(self, name: str):
        '''
        get coordinate of name
        :param name:
        :return: (latitude, longitude)
        '''
        result = self.data[self.data['shopname'] == name]
        if len(result) == 0:
            return None
        return (result.iloc[0].to_dict()['latitude'], result.iloc[0].to_dict()['longitude'])

    def get_one_shopname(self):
        return self.data.iloc[random.randint(0,len(self.data) - 1)]["shopname"]

    def get_one_type(self):
        return self.data.iloc[random.randint(0,len(self.data) - 1)]["type"]

    def run_for_distance(self,
                         name1: str, name2: str
                         ) -> float:
        """
        Search for distance of attractions (km).
        Return 0 if the name is not found.
        """

        result1 = self.data[self.data['shopname'] == name1]
        result2 = self.data[self.data['shopname'] == name2]
        if len(result1) == 0:
            result1 = self.data[self.data['shopname'].str.contains(name1, case=False, na=False)]
        if len(result2) == 0:
            result2 = self.data[self.data['shopname'].str.contains(name2, case=False, na=False)]
        if len(result1) == 0:
            print(f"There is no {name1} in this city.")
            return 0
        if len(result2) == 0:
            print(f"There is no {name2} in this city.")
            return 0
        if len(result1) == 0 or len(result2) == 0:
            print(f"There is no {name1} or {name2} in this city.")
            return 0
        longitude1 = result1.sort_values(by=["star"], ascending=False).iloc[0].to_dict()["longitude"]
        longitude2 = result2.sort_values(by=["star"], ascending=False).iloc[0].to_dict()["longitude"]
        latitude1 = result1.sort_values(by=["star"], ascending=False).iloc[0].to_dict()["latitude"]
        latitude2 = result2.sort_values(by=["star"], ascending=False).iloc[0].to_dict()["latitude"]

        return calculate_distance((latitude1, longitude1), (latitude2, longitude2))


class Attractions(Tools):
    def __init__(self, path="database/new_attraction.csv"):
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
    def __init__(self, path="database/new_restaurant.csv"):
        super().__init__(path)
        print("Restaurants loaded.")

    def run(self,
            restaurant_name: str,
            ) -> DataFrame:
        """Search for restaurant ."""

        return super().run(restaurant_name)

    def run_for_distance(self,
            restaurant_name1: str, restaurant_name2: str
            ) -> float:
        """Search for restaurant ."""
        return super().run_for_distance(restaurant_name1, restaurant_name2)





if __name__ == '__main__':
    attractions = Attractions()

    print(attractions.run("西湖"))
    print(attractions.run_for_distance("西湖", "风景区"))
    print(attractions.get_nearest_restaurants(30.26090127, 120.1470172, 10))
    print(attractions.get_one_shopname())