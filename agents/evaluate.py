import os, sys
import random
import re, json

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "./")))
from tools.apis import Restaurants, Attractions

class Evaluator:
    def __init__(self):
        self.restaurant = Restaurants()
        self.attraction = Attractions()
        self.queries = []
        self.answers = []
        self.results = []
        self.step = 0

        self.normal_num = 0
        self.complete_num = 0
        self.attraction_num = 0
        self.restaurant_num = 0
        self.budget_num = 0
        self.all_num = 0


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
            budget = people * random.randint(200,300)
            query = self.generate_query(want_restaurant, want_attraction, restaurant_name, attraction_name, people, budget)
            self.queries.append(query)
            self.answers.append({"want_restaurant": want_restaurant, "want_attraction": want_attraction, "restaurant_name": restaurant_name, "attraction_name": attraction_name, "people": people, "budget": budget})
            print(f'旅游路线：{query}')
        return self.queries
    def generate_query(self, want_restaurant, want_attraction, restaurant_name, attraction_name, people, budget):
        return f"请帮我规划一条在杭州的一日旅游路线，{want_attraction}想要去{attraction_name}游玩，{want_restaurant}想要吃{restaurant_name}类型的美食，一共有{people}个人，总共预算需要在{budget}元以内。"

    def evaluate(self, agent_output):
        answer = self.answers[self.step]
        self.step += 1
        all_flag = True
        print(agent_output)
        try:
            match = re.search(r'(\{.*\})', agent_output, re.DOTALL)
            if match:
                extracted_json = match.group(1)
                data = json.loads(extracted_json)
                attraction1 = data['上午景点']
                lunch = data['午餐']
                attraction2 = data['下午景点']
                dinner = data['晚餐']
                print(data)
                print(f"attraction1 {attraction1}, attraction2 {attraction2}, lunch {lunch}, dinner {dinner}.")
                self.normal_num += 1
                if self.eval_complete(attraction1, lunch, attraction2, dinner):
                    self.complete_num += 1
                else:
                    all_flag = False

                if self.eval_attraction(answer, data['上午景点'], data['下午景点']):
                    self.attraction_num += 1
                else:
                    all_flag = False

                if self.eval_restaurant(answer, data['午餐'], data['晚餐']):
                    self.restaurant_num += 1
                else:
                    all_flag = False

                if self.eval_budget(answer, data['午餐'], data['晚餐']):
                    self.budget_num += 1
                else:
                    all_flag = False

                self.all_num += 1 if all_flag else 0
            else:
                print("output wrong json")
        except Exception as e:
            print(e)
            print(f'Step {self.step} result not completed')

    def eval_complete(self, attraction1, lunch, attraction2, dinner):
        if self.attraction.run(attraction1) == f"There is no {attraction1} in this city.":
            return False
        if self.restaurant.run(lunch) == f"There is no {lunch} in this city.":
            return False
        if self.attraction.run(attraction2) == f"There is no {attraction2} in this city.":
            return False
        if self.restaurant.run(dinner) == f"There is no {dinner} in this city.":
            return False
        return True

    def eval_attraction(self, answer, attraction1, attraction2):
        if answer["want_attraction"] == '':
            return attraction1 == answer["attraction_name"] or attraction2 == answer["attraction_name"]
        else:
            return attraction1 != answer["attraction_name"] and attraction2 != answer["attraction_name"]

    def eval_restaurant(self, answer, restaurant1, restaurant2):
        if answer["want_restaurant"] == '':
            return self.restaurant.run(restaurant1)['type'] == answer["restaurant_name"] or self.restaurant.run(restaurant2)['type'] == answer["restaurant_name"]
        else:
            return self.restaurant.run(restaurant1)['type'] != answer["restaurant_name"] and self.restaurant.run(restaurant2)['type'] != answer["restaurant_name"]

    def eval_budget(self, answer, restaurant1, restaurant2):
        price1 = self.restaurant.run(restaurant1)["price"]
        price2 = self.restaurant.run(restaurant2)["price"]
        people = answer["people"]
        budget = answer["budget"]

        return (price1 + price2) * people <= budget

    def print_result(self):
        print(f"normal: {self.normal_num}")
        print(f"complete: {self.complete_num}")
        print(f"attraction: {self.attraction_num}")
        print(f"restaurant: {self.restaurant_num}")
        print(f"budget: {self.budget_num}")
        print(f"all: {self.all_num}")


if __name__ == '__main__':
    restaurant = Restaurants()
    attraction = Attractions()
    data = {'交通': '地铁', '早餐': '杭州西子湖四季酒店-WL BISTRO西湖餐厅(灵隐路店)', '上午景点': '盖叫天故居', '午餐': '弄堂里·江南名肴(西湖龙井路店)', '下午景点': '灵隐寺', '晚餐': '山外山菜馆(玉泉路店)', '晚上的景点': '西湖'}
    attraction1 = data['上午景点']
    lunch = data['午餐']
    attraction2 = data['下午景点']
    dinner = data['晚餐']
    if attraction.run(attraction1) == f"There is no {attraction1} in this city.":
        print("attraction1 not exist")
    if restaurant.run(lunch) == f"There is no {lunch} in this city.":
        print("lunch not exist")
    if attraction.run(attraction2) == f"There is no {attraction2} in this city.":
        print("attraction2 not exist")
    if restaurant.run(dinner) == f"There is no {dinner} in this city.":
        print("dinner not exist")