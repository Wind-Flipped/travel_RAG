import json
import random
import re
import pandas as pd
from zhipuai import ZhipuAI
from tqdm import tqdm
from poi2route import Poi2Route

tool = Poi2Route()

with open("/home/wangb/cyo/graduation/rag/databases/xihu_1/document.json", "r", encoding="utf-8") as f1:
    data1 = json.load(f1)

request = []

model = ZhipuAI(api_key="8cf93821658b7df312645b6dc443b871.aY8DFiq17G0NrVmx")


def extract_route(item):
    # 匹配“路线:”后面的地点，多个地点以逗号分隔
    match = re.search(r"路线[:：](.+)帖子", item)
    if match:
        # 提取并按逗号分隔，过滤掉空结果
        return [place.strip() for place in match.group(1).split(",") if place.strip()]
    return []


def extract_key(item):
    match = re.search(r"关键词[:：](.+?)路线", item)
    if match:
        # 提取并按逗号分隔，过滤掉空结果
        return [place.strip() for place in match.group(1).split(",") if place.strip()]
    return []


# for data in tqdm(data1):
#     route = extract_route(data)
#     key_word = extract_key(data)
#     target_places = random.sample(route, 2)
#     input_txt = "请帮我推荐一条西湖一日游的旅游路线，要求是：" + '，'.join(key_word) + '。除此，我还希望去' + '，'.join(
#         target_places) + '游玩。'
#     print(input_txt)
#     ai_input = model.chat.completions.create(
#         model="glm-4-air",
#         messages=[
#             {"role": "user",
#              "content": "请改写下面的文本，使之更像人提出的需求，不要包含除文本中提及的地点之外的地点。只输出结果。文本是：" + input_txt},
#         ],
#     )
#     request.append({
#         "input": input_txt,
#         "ai_input": ai_input.choices[0].message.content,
#         "route": route,
#         "target_place": target_places
#     })
#
# output_file = "/home/wangb/cyo/graduation/rag/databases/xihu_1/key_place2_requests.json"
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(request, f, ensure_ascii=False, indent=4)

for data in tqdm(data1):
    route = extract_route(data)
    key_word = extract_key(data)
    target_places = random.sample(route, 2)
    time_route, time_cost = tool.cost(route, "time")
    distance_route, distance_cost = tool.cost(route, "distance")
    input_txt = "请帮我推荐一条西湖一日游的旅游路线，要求是：" + '，'.join(key_word) + '。除此，我还希望去' + '，'.join(
        target_places) + '游玩。' + f'我希望我的交通时间在{int(time_cost)}分钟左右，交通距离在{int(distance_cost)}千米左右。'
    print(input_txt)
    ai_input = model.chat.completions.create(
        model="glm-4-air",
        messages=[
            {"role": "user",
             "content": "请改写下面的文本，使之更像人提出的需求，不要包含除文本中提及的地点之外的地点。只输出结果。文本是：" + input_txt},
        ],
    )
    request.append({
        "input": input_txt,
        "ai_input": ai_input.choices[0].message.content,
        "route": route,
        "target_place": target_places,
        "distance": distance_cost,
        "time": time_cost
    })

output_file = "/home/wangb/cyo/graduation/rag/databases/xihu_1/key_place2_td_requests.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(request, f, ensure_ascii=False, indent=4)
