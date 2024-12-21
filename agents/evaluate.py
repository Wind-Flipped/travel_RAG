import os, sys
import random
import re, json
import numpy as np
import json
import math
from llms import LLMs

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "./")))
from tools.apis import Restaurants, Attractions
from util.calculate import calculate_distance
from rag.component.embedding import Zhipuembedding
from zhipuai import ZhipuAI
from prompts import evaluate_request_route_zh



class Evaluator:
    def __init__(self, have_truth = False, api_key = "c59db5e044cd9cd453a49b462a659697.RD2fEoEAwM5EhPuE"):
        self.restaurant = Restaurants()
        self.attraction = Attractions()
        self.queries = []
        self.answers = []
        self.results = []
        self.step = 0
        self.valid_restaurant = True
        self.valid_attraction = True

        self.normal_num = 0
        self.complete_num = 0
        self.attraction_num = 0
        self.restaurant_num = 0
        self.budget_num = 0
        self.all_num = 0

        self.avg_distance = 0
        self.valid_distance = 0
        self.avg_score = 0
        self.valid_score = 0
        self.llm = LLMs()
        self.eval_log = []

        if have_truth:
            self.embedding_model = Zhipuembedding(api_key=api_key)

            with open("/home/wangb/cyo/graduation/rag/databases/popularity.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.popularity = {eval(key): value for key, value in data.items()}

            with open('/home/wangb/cyo/graduation/rag/databases/hangzhou_poi_coo2.json', 'r', encoding='utf-8') as file:
                poi_file = json.load(file)
            self.poi_dict = {poi['name']: poi for poi in poi_file}

            self.jaccard_similarity = 0
            self.match_similarity = 0
            self.center_distance = 0
            self.distance_similarity = 0
            self.request2route = 0
            self.popularity_similarity = 0


    def generate_request(self, number : int = 50):
        while number > 0:
            number -= 1
            restaurant_name = self.restaurant.get_one_type()
            attraction_name = self.attraction.get_one_shopname()
            random_number1 = random.uniform(0,1)
            random_number2 = random.uniform(0,1)
            want_attraction = "" if random_number1 > 0.5 else "不"
            want_restaurant = "" if random_number2 > 0.5 else "不"
            people = random.randint(1,6)
            budget = people * random.randint(150,300)
            query = self.generate_query(want_restaurant, want_attraction, restaurant_name, attraction_name, people, budget)
            self.queries.append(query)
            self.answers.append({"want_restaurant": want_restaurant, "want_attraction": want_attraction, "restaurant_name": restaurant_name, "attraction_name": attraction_name, "people": people, "budget": budget})
            print(f'旅游路线：{query}')
        return self.queries
    def generate_query(self, want_restaurant, want_attraction, restaurant_name, attraction_name, people, budget):
        return f"请帮我规划一条在杭州的一日旅游路线，{want_attraction}想要去{attraction_name}游玩，{want_restaurant}想要吃{restaurant_name}类型的美食，一共有{people}个人，总共预算需要在{budget}元以内。"

    def evaluate_real(self, agent_output, target_place, query, truth):
        self.eval_log.append({"step": self.step, "normal": False,
                              "complete": False, "attraction": False, "restaurant": False, "budget": False,
                              "avg_distance": 0, "avg_score": 0,
                              "jaccard_similarity": 0, "exact_match_similarity": 0, "center_distance": 0,
                              "distance_similarity": 0, "request2route": 0, "popularity_similarity": 0})
        self.step += 1
        all_flag = True
        self.valid_restaurant = True
        self.valid_attraction = True
        answer = {"want_attraction": "", "attraction_name": target_place[0]}
        try:
            match = re.search(r'(\{.*\})', agent_output, re.DOTALL)
            if match:
                extracted_json = match.group(1)
                data = json.loads(extracted_json)
                attraction1 = data['上午景点']
                attraction1 = [item.strip() for item in attraction1.split(",")]
                lunch = data['午餐']
                attraction2 = data['下午景点']
                attraction2 = [item.strip() for item in attraction2.split(",")]
                dinner = data['晚餐']
                attraction3 = data['晚上景点']
                attraction3 = [item.strip() for item in attraction3.split(",")]
                print(data)
                attraction_list = attraction1 + attraction2 + attraction3
                attraction_list = [item for item in attraction_list if item != "-" and item != ""]
                print(f"attraction_list {attraction_list}, lunch {lunch}, dinner {dinner}.")
                self.normal_num += 1
                self.eval_log[-1]["normal"] = True
                if self.eval_complete(attraction_list, "金猪", "金猪"):
                    self.eval_log[-1]["complete"] = True
                    self.complete_num += 1
                    self.eval_distance(attraction_list)
                else:
                    all_flag = False

                if self.valid_attraction and self.eval_attraction(answer, attraction_list):
                    self.eval_log[-1]["attraction"] = True
                    self.attraction_num += 1
                else:
                    all_flag = False

                # if self.valid_restaurant and self.eval_restaurant(answer, lunch, dinner):
                #     self.eval_log[-1]["restaurant"] = True
                #     self.restaurant_num += 1
                # else:
                #     all_flag = False

                # if self.eval_budget(answer, lunch, dinner):
                #     self.eval_log[-1]["budget"] = True
                #     self.budget_num += 1
                # else:
                #     all_flag = False

                self.all_num += 1 if all_flag else 0
                self.calculate_ai_similarity(attraction_list, query)

                score = self.calculate_jaccard_similarity(attraction_list, truth)
                self.jaccard_similarity += score
                self.eval_log[-1]["jaccard_similarity"] = score

                score = self.calculate_exact_match_similarity(attraction_list, truth)
                self.match_similarity += score
                self.eval_log[-1]["exact_match_similarity"] = score

                score = self.calculate_distance_similarity(attraction_list, truth)
                self.distance_similarity += score
                self.eval_log[-1]["distance_similarity"] = score

                score = self.calculate_popularity_similarity(attraction_list, truth)
                self.popularity_similarity += score
                self.eval_log[-1]["popularity_similarity"] = score

                score = self.calculate_request2route(attraction_list, truth)
                self.request2route += score
                self.eval_log[-1]["request2route"] = score

                score = self.calculate_center_distance(attraction_list, truth)
                self.center_distance += score
                self.eval_log[-1]["center_distance"] = score

            else:
                print("output wrong json")
        except Exception as e:
            print(e)
            print(f'Step {self.step} result not completed')

    def print_real_result(self, mode):
        print(f"normal: {self.normal_num}")
        print(f"complete: {self.complete_num}")
        print(f"attraction: {self.attraction_num}")
        print(f"restaurant: {self.restaurant_num}")
        print(f"budget: {self.budget_num}")
        print(f"all: {self.all_num}")

        print(f"avg distance: {self.avg_distance / self.valid_distance if self.valid_distance > 0 else None}")
        print(f"avg score: {self.avg_score / self.valid_score if self.valid_score > 0 else None}")

        print(f"avg jaccard_similarity (+): {self.jaccard_similarity / self.normal_num}")
        print(f"avg exact_match_similarity (+): {self.match_similarity / self.normal_num}")
        print(f"avg distance_similarity (-): {self.distance_similarity / self.normal_num}")
        print(f"avg request2route (+): {self.request2route / self.normal_num}")
        print(f"avg popularity_similarity (-): {self.popularity_similarity / self.normal_num}")
        print(f"avg center_distance (-): {self.center_distance / self.normal_num}")

        self.eval_log.append(
            {"normal": self.normal_num, "complete": self.complete_num, "attraction": self.attraction_num,
             "restaurant": self.restaurant_num, "budget": self.budget_num, "all": self.all_num,
             "avg_distance": self.avg_distance / self.valid_distance if self.valid_distance > 0 else None,
             "avg_score": self.avg_score / self.valid_score if self.valid_score > 0 else None,
             "jaccard_similarity": self.jaccard_similarity / self.normal_num, "exact_match_similarity": self.match_similarity / self.normal_num,
             "distance_similarity": self.distance_similarity / self.normal_num, "request2route": self.request2route / self.normal_num,
             "popularity_similarity": self.popularity_similarity / self.normal_num, "center_distance": self.center_distance / self.normal_num})
        # save the results
        if not os.path.exists(f'./logs/{mode}'):
            os.makedirs(f'./logs/{mode}')
        with open(os.path.join(f'./logs/{mode}/real_eval.json'), 'w') as f:
            json.dump(self.eval_log, f, indent=4, ensure_ascii=False)


    def evaluate(self, agent_output):
        self.eval_log.append({"step": self.step, "normal": False,
                              "complete": False, "attraction": False, "restaurant": False, "budget": False,
                              "avg_distance": 0, "avg_score": 0})
        answer = self.answers[self.step]
        query = self.queries[self.step]
        self.step += 1
        all_flag = True
        self.valid_restaurant = True
        self.valid_attraction = True
        try:
            match = re.search(r'(\{.*\})', agent_output, re.DOTALL)
            if match:
                extracted_json = match.group(1)
                data = json.loads(extracted_json)
                attraction1 = data['上午景点']
                attraction1 = [item.strip() for item in attraction1.split(",")]
                lunch = data['午餐']
                attraction2 = data['下午景点']
                attraction2 = [item.strip() for item in attraction2.split(",")]
                dinner = data['晚餐']
                attraction3 = data['晚上景点']
                attraction3 = [item.strip() for item in attraction3.split(",")]
                print(data)
                attraction_list = attraction1 + attraction2 + attraction3
                attraction_list = [item for item in attraction_list if item != "-" and item != ""]
                print(f"attraction_list {attraction_list}, lunch {lunch}, dinner {dinner}.")
                self.normal_num += 1
                self.eval_log[-1]["normal"] = True
                if self.eval_complete(attraction_list, lunch, dinner):
                    self.eval_log[-1]["complete"] = True
                    self.complete_num += 1
                    self.eval_distance(attraction_list)
                else:
                    all_flag = False

                if self.valid_attraction and self.eval_attraction(answer, attraction_list):
                    self.eval_log[-1]["attraction"] = True
                    self.attraction_num += 1
                else:
                    all_flag = False

                if self.valid_restaurant and self.eval_restaurant(answer, lunch, dinner):
                    self.eval_log[-1]["restaurant"] = True
                    self.restaurant_num += 1
                else:
                    all_flag = False

                if self.eval_budget(answer, lunch, dinner):
                    self.eval_log[-1]["budget"] = True
                    self.budget_num += 1
                else:
                    all_flag = False

                self.all_num += 1 if all_flag else 0
                self.calculate_ai_similarity(attraction_list, query)
            else:
                print("output wrong json")
        except Exception as e:
            print(e)
            print(f'Step {self.step} result not completed')

    def eval_complete(self, attraction_list, lunch, dinner):
        for attraction in attraction_list:
            if self.attraction.run(attraction) == f"There is no {attraction} in this city.":
                self.valid_attraction = False
                return False

        if self.restaurant.run(lunch) == f"There is no {lunch} in this city.":
            self.valid_restaurant = False
            return False
        if self.restaurant.run(dinner) == f"There is no {dinner} in this city.":
            self.valid_restaurant = False
            return False
        return True

    def eval_distance(self, attraction_list):
        try:
            length = len(attraction_list)
            sum_distance = 0
            for i in range(length - 1):
                coord1 = self.attraction.get_coordinate(attraction_list[i])
                coord2 = self.attraction.get_coordinate(attraction_list[i + 1])
                distance = calculate_distance(coord1, coord2)
                sum_distance += distance
            self.avg_distance += sum_distance / length
            self.valid_distance += 1
            self.eval_log[-1]["avg_distance"] = sum_distance / length
        except Exception as e:
            print(e)
            print("eval_distance error")

    def eval_attraction(self, answer, attraction_list):
        if answer["want_attraction"] == '':
            return answer["attraction_name"] in attraction_list
        else:
            return answer["attraction_name"] not in attraction_list

    def eval_restaurant(self, answer, restaurant1, restaurant2):
        if answer["want_restaurant"] == '':
            return self.restaurant.run(restaurant1)['type'] == answer["restaurant_name"] or self.restaurant.run(restaurant2)['type'] == answer["restaurant_name"]
        else:
            return self.restaurant.run(restaurant1)['type'] != answer["restaurant_name"] and self.restaurant.run(restaurant2)['type'] != answer["restaurant_name"]

    def eval_budget(self, answer, restaurant1, restaurant2):
        price1 = self.restaurant.run(restaurant1)["price"]
        price2 = self.restaurant.run(restaurant2)["price"]
        match = re.search(r'(\d+)', price1)
        extracted_price1 = int(match.group(1))
        match = re.search(r'(\d+)', price2)
        extracted_price2 = int(match.group(1))
        people = answer["people"]
        budget = answer["budget"]

        return (extracted_price1 + extracted_price2) * people <= budget

    def print_result(self):
        print(f"normal: {self.normal_num}")
        print(f"complete: {self.complete_num}")
        print(f"attraction: {self.attraction_num}")
        print(f"restaurant: {self.restaurant_num}")
        print(f"budget: {self.budget_num}")
        print(f"all: {self.all_num}")

        print(f"avg distance: {self.avg_distance / self.valid_distance if self.valid_distance > 0 else None}")
        print(f"avg score: {self.avg_score / self.valid_score if self.valid_score > 0 else None}")


        self.eval_log.append({"normal": self.normal_num, "complete": self.complete_num, "attraction": self.attraction_num,
                              "restaurant": self.restaurant_num, "budget": self.budget_num, "all": self.all_num,
                              "avg_distance": self.avg_distance / self.valid_distance if self.valid_distance > 0 else None,
                              "avg_score": self.avg_score / self.valid_score if self.valid_score > 0 else None})
        # save the results
        with open(os.path.join(f'./logs/eval.json'), 'w') as f:
            json.dump(self.eval_log, f, indent=4, ensure_ascii=False)


    def calculate_jaccard_similarity(self, route, truth):
        set1, set2 = set(route), set(truth)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union


    def calculate_exact_match_similarity(self, route, truth):
        intersection = len(set(route) & set(truth))
        return intersection / len(set(truth))


    def get_poi_vector(self, poi):
        poi = self.poi_dict.get(poi)
        if poi:
            return poi['embedding']
        else:
            raise ValueError(f"ERROR: Wrong {poi}, can not get embedding")


    def get_poi_description(self, poi):
        poi = self.poi_dict.get(poi)
        if poi:
            return poi['summarized_description']
        else:
            raise ValueError(f"ERROR: Wrong {poi}, can not get description")


    def get_poi_location(self, poi):
        poi = self.poi_dict.get(poi)
        if poi:
            return poi['longitude'], poi['latitude']
        else:
            raise ValueError(f"ERROR: Wrong {poi}, can not get location")


    def aggregate_poi_center(self, poi_list):
        total_lat = 0
        total_lon = 0

        valid_poi_num = 0

        for poi in poi_list:
            try:
                lon, lat = self.get_poi_location(poi)
                total_lat += lat
                total_lon += lon
                valid_poi_num += 1
            except ValueError as v:
                print(v)

        # 重心的经纬度
        center_lat = total_lat / valid_poi_num
        center_lon = total_lon / valid_poi_num
        return center_lon, center_lat


    def aggregate_poi_distances(self, poi_list):
        total_distance = 0
        for i in range(len(poi_list) - 1):
            try:
                poi = poi_list[i]
                poi1 = poi_list[i + 1]
                lon, lat = self.get_poi_location(poi)
                lon1, lat1 = self.get_poi_location(poi1)
                distance = calculate_distance((lat1, lon1), (lat, lon))
                total_distance += distance
            except ValueError as v:
                print(v)

        return total_distance


    def aggregate_poi_popularity(self, poi_list):
        total_popularity = 0
        for i in range(len(poi_list) - 1):
            try:
                poi = poi_list[i]
                poi1 = poi_list[i + 1]
                cur_popularity = self.popularity.get((poi, poi1), 0)
                total_popularity += cur_popularity
            except ValueError as v:
                print(v)

        return total_popularity


    def calculate_cosine_similarity(self, route, truth):

        return None


    def calculate_center_distance(self, route, truth):
        center1_lon, center1_lat = self.aggregate_poi_center(route)
        center2_lon, center2_lat = self.aggregate_poi_center(truth)
        return calculate_distance((center1_lat, center1_lon), (center2_lat, center2_lon))


    def calculate_distance_similarity(self, route, truth):
        distance1 = self.aggregate_poi_distances(route)
        distance2 = self.aggregate_poi_distances(truth)
        distance_diff = abs(distance1 - distance2)
        if distance2 != 0:
            distance_similarity = distance_diff / distance2
            return distance_similarity
        else:
            print(f'truth {truth} has no distance')
            return 0


    def calculate_request2route(self, request, route):
        request_vector = self.embedding_model.get_embedding(request)
        dots = 0
        valid_pois = 0
        for poi in route:
            try:
                poi_embedding = self.get_poi_vector(poi)
                dot_product = np.dot(request_vector, poi_embedding)
                dots += dot_product
                valid_pois += 1
            except ValueError as v:
                print(v)
        return dots / valid_pois


    def calculate_popularity_similarity(self, route, truth):
        popularity1 = self.aggregate_poi_popularity(route)
        popularity2 = self.aggregate_poi_popularity(truth)
        popularity_diff = abs(popularity1 - popularity2)
        if popularity2 != 0:
            popularity_similarity = popularity_diff / popularity2
            return popularity_similarity
        else:
            print(f'truth {truth} has no popularity')
            return 0


    def calculate_ai_similarity(self, route, request):
        pois_description = []
        for poi in route:
            try:
                description = self.get_poi_description(poi)
            except Exception as e:
                description = e
            pois_description.append(description)
        format_poi_info = "\n".join(f"-第{index + 1}条景点信息: {item}" for index, item in enumerate(pois_description))
        template = evaluate_request_route_zh.format(
                request=request,
                route=route,
                poi_info=format_poi_info)
        response = self.llm(template)
        match = re.search(r'(\d+)', response)
        if match:
            score = int(match.group(1))
            self.avg_score += score
            self.valid_score += 1
            self.eval_log[-1]["avg_score"] = score




if __name__ == '__main__':
    restaurant = Restaurants()
    attraction = Attractions()
    data = ""
    # attraction1 = data['上午景点']
    # lunch = data['午餐']
    # attraction2 = data['下午景点']
    # dinner = data['晚餐']
    # price1 = restaurant.run(lunch)["price"]
    # match = re.search(r'(\d+)', price1)
    # if match:
    #     extracted_price1 = int(match.group(1))
    # price2 = restaurant.run(dinner)["price"]
    # match = re.search(r'(\d+)', price2)
    # if match:
    #     extracted_price2 = int(match.group(1))
    # people = 6
    # budget = 1716
    # print((extracted_price1 + extracted_price2) * people <= budget)