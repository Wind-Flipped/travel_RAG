import json
import os, sys
from zhipuai import ZhipuAI
import re, random
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "./")))
from tools.apis import Restaurants, Attractions
from util.calculate import calculate_distance
# data_path = "../MediaCrawler-main/MediaCrawler-main/data/xhs/json/"

class GenerateRequest:
    def __init__(self, have_truth = False, api_key = "c59db5e044cd9cd453a49b462a659697.RD2fEoEAwM5EhPuE"):
        self.restaurant = Restaurants()
        self.attraction = Attractions()
        self.step = 0

    def generate_request(self, number : int = 50):
        base_request = []
        i = 1
        while i <= number:

            restaurant_name = self.restaurant.get_one_type()
            attraction_name = self.attraction.get_one_shopname()
            random_number1 = random.uniform(0,1)
            random_number2 = random.uniform(0,1)
            want_attraction = "" if random_number1 > 0.25 else "不"
            want_restaurant = "" if random_number2 > 0.75 else "不"
            people = random.randint(1,6)
            budget = people * random.randint(150,300)
            distance = random.randint(20, 100)
            query = self.generate_query(want_restaurant, want_attraction, restaurant_name, attraction_name, people, budget, distance)
            print(f'旅游路线：{query}')
            base_request.append({"index": i, "query": query, "want_restaurant": want_restaurant, "want_attraction": want_attraction, "restaurant_name": restaurant_name, "attraction_name": attraction_name, "people": people, "budget": budget, "distance": distance})
            i += 1
        if not os.path.exists("data"):
            os.makedirs("data")
        with open("data/base_request1.json", 'w', encoding='utf-8') as f:
            json.dump(base_request, f, ensure_ascii=False, indent=4)

    def generate_query(self, want_restaurant, want_attraction, restaurant_name, attraction_name, people, budget, distance):
        return f"请帮我规划一条在杭州的一日旅游路线，{want_attraction}想要去{attraction_name}游玩，{want_restaurant}想要吃{restaurant_name}类型的美食，按照顺序参观不同景点之间的交通距离之和不要超过{distance}km，一共有{people}个人，总共预算需要在{budget}元以内。"



def get_rawdata():
    input_file = "/home/wangb/cyo/graduation/rag/databases/hangzhou/document.json"
    output_file = "./output/data.json"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    responses = set()

    client = ZhipuAI(api_key="c59db5e044cd9cd453a49b462a659697.RD2fEoEAwM5EhPuE")

    for index, note in enumerate(data):
        try:
            message_content = note
            response_route = client.chat.completions.create(
                model="glm-4-air",
                messages=[
                    {"role": "user",
                     "content": "请从下面的帖子中提取出不同的景点，每个景点需要带有引号，不同景点用', '分割，最终只返回一个景点列表，用中括号'['']'表示出来，使得答案能被解析为一个列表。帖子的内容是：" + message_content},
                ],
            )
            answer = response_route.choices[0].message.content
            print(index)
            print(answer)
            answer = eval(answer)
            for item in answer:
                responses.add(item)
            print(responses)


        except Exception as e:
            continue

    # valid_responses = []
    # for response in responses:
    #     answer = response['route']
    #     if re.match(r'^[\u4e00-\u9fa5\w\s,]+$', answer):
    #         parts = answer.split(',')
    #         if all(part.strip() != "杭州" for part in parts) and len(parts) >= 3:
    #             valid_responses.append(response)
    #             print(answer)

    # 将结果保存到新的文件
    responses = list(responses)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=4)

    print(f"回答已保存到 {output_file} 文件中。")


def get_route_data():
    input_file1 = "./output/data.json"
    input_file2 = "/home/wangb/cyo/graduation/rag/databases/hangzhou_poi_coo2.json"

    with open(input_file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)

    with open(input_file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)

    extract_data = [item["name"] for item in data2]

    empty_data = []
    for item in data1:
        if item not in extract_data:
            empty_data.append(item)

    print(empty_data)
    with open("./output_file.json", 'w', encoding='utf-8') as f:
        json.dump(empty_data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    gen_request = GenerateRequest()
    gen_request.generate_request(100)

